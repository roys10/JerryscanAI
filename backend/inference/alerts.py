import smtplib
from email.message import EmailMessage
import requests
import ssl
from .config import ConfigManager

class AlertManager:
    """Tracks inspection failures and triggers alerts when thresholds are reached."""
    
    def __init__(self, config_manager: ConfigManager, history_manager=None):
        self.config = config_manager
        self.history_manager = history_manager
        self.consecutive_failures = 0
        self.alert_sent_for_current_streak = False
        self.pass_rate_alert_active = False

    def evaluate_session(self, overall_status: str, session_id: str):
        """Processes a new inspection result."""
        
        # 1. Handle Consecutive Failures
        if overall_status == "FAIL":
            self.consecutive_failures += 1
            print(f"[AlertManager] Failure {self.consecutive_failures} recorded for session {session_id}.")
            
            threshold = int(self.config.get("alert_threshold", 3))
            
            if self.consecutive_failures >= threshold and not self.alert_sent_for_current_streak:
                print(f"[AlertManager] Streak Threshold ({threshold}) reached! Triggering alerts...")
                self.trigger_streak_alert(session_id, self.consecutive_failures)
                self.alert_sent_for_current_streak = True
                
        elif overall_status == "PASS":
            if self.consecutive_failures > 0:
                print(f"[AlertManager] System recovered. Resetting failure streak (was {self.consecutive_failures}).")
            self.consecutive_failures = 0
            self.alert_sent_for_current_streak = False

        # 2. Handle Pass Rate Threshold
        if self.history_manager:
            window = int(self.config.get("alert_pass_rate_window", 50))
            threshold = float(self.config.get("alert_pass_rate_threshold", 90))
            
            # Fetch recent history
            recent = self.history_manager.get_history(limit=window)
            if len(recent) >= 5: # Minimum sample size to avoid jitter
                passes = len([s for s in recent if s["overall_status"] == "PASS"])
                current_rate = (passes / len(recent)) * 100
                
                if current_rate < threshold:
                    if not self.pass_rate_alert_active:
                        print(f"[AlertManager] Pass Rate Drop! {current_rate:.1f}% < {threshold}% (Window: {len(recent)})")
                        self.trigger_pass_rate_alert(current_rate, threshold, len(recent))
                        self.pass_rate_alert_active = True
                else:
                    if self.pass_rate_alert_active:
                        print(f"[AlertManager] Pass Rate recovered to {current_rate:.1f}%.")
                    self.pass_rate_alert_active = False

    def trigger_streak_alert(self, session_id: str, count: int):
        """Dispatches streak-based alerts."""
        subject = f"🚨 JerryScan AI: {count} Consecutive Failures!"
        body = f"The visual inspection system has detected {count} consecutive defective products.\n\nLatest Session ID: {session_id}\nPlease check the production line immediately."
        self._dispatch(subject, body)

    def trigger_pass_rate_alert(self, current_rate: float, threshold: float, window: int):
        """Dispatches pass-rate-based alerts."""
        subject = f"⚠️ JerryScan AI: Pass Rate Drop ({current_rate:.1f}%)"
        body = f"The rolling pass rate has dropped below your threshold of {threshold}%.\n\nCurrent Rate: {current_rate:.1f}%\nWindow Size: {window} samples\n\nQuality may be degrading. Please investigate."
        self._dispatch(subject, body)

    def _dispatch(self, subject: str, body: str):
        """Internal helper to send to all channels."""
        # 1. Email
        target_email = self.config.get("alert_email_address")
        if target_email: self._send_email(target_email, subject, body)
            
        # 2. Webhook
        webhook_url = self.config.get("alert_webhook_url")
        if webhook_url: self._send_webhook(webhook_url, subject, body)

    def _send_email(self, to_email: str, subject: str, body: str):
        try:
            smtp_server = self.config.get("smtp_server")
            smtp_port = int(self.config.get("smtp_port"))
            smtp_user = self.config.get("smtp_user")
            smtp_password = self.config.get("smtp_password")

            if not all([smtp_server, smtp_port, smtp_user, smtp_password]):
                print("[AlertManager] SMTP configuration incomplete. Skipping email alert.")
                return

            msg = EmailMessage()
            msg.set_content(body)
            msg['Subject'] = subject
            msg['From'] = smtp_user
            msg['To'] = to_email

            if smtp_port == 465:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
                    server.login(smtp_user, smtp_password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                    server.send_message(msg)
            print("[AlertManager] Email sent successfully.")
        except Exception as e:
            print(f"[AlertManager] Failed to send email: {e}")

    def _send_webhook(self, url: str, subject: str, body: str):
        try:
            payload = {"text": f"*{subject}*\n{body}", "subject": subject, "body": body}
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            print("[AlertManager] Webhook dispatched successfully.")
        except Exception as e:
            print(f"[AlertManager] Failed to dispatch webhook: {e}")
