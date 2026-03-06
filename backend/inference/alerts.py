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
        
        # In-memory tracking for rule states (streaks, active alerts)
        # Format: { rule_id: { "streak": int, "alert_active": bool } }
        self.rule_states = {}

    def _get_rule_state(self, rule_id: str):
        if rule_id not in self.rule_states:
            self.rule_states[rule_id] = {"streak": 0, "alert_active": False}
        return self.rule_states[rule_id]

    def evaluate_session(self, overall_status: str, session_id: str):
        """Processes a new inspection result against all active rules."""
        rules = self.config.get("alerts", [])
        
        for rule in rules:
            if not rule.get("enabled", True):
                continue
            
            rule_id = rule.get("id")
            rule_type = rule.get("type")
            state = self._get_rule_state(rule_id)
            
            # --- EVALUATE RULE ---
            triggered = False
            details = ""

            if rule_type == "consecutive_fails":
                if overall_status == "FAIL":
                    state["streak"] += 1
                    threshold = int(rule.get("threshold", 3))
                    if state["streak"] >= threshold and not state["alert_active"]:
                        triggered = True
                        details = f"{state['streak']} consecutive failures detected."
                else:
                    state["streak"] = 0
                    state["alert_active"] = False # Recovered

            elif rule_type == "pass_rate" and self.history_manager:
                window = int(rule.get("window", 50))
                threshold = float(rule.get("threshold", 90))
                
                recent = self.history_manager.get_history(limit=window)
                if len(recent) >= 5: # Min samples to avoid noise
                    passes = len([s for s in recent if s["overall_status"] == "PASS"])
                    rate = (passes / len(recent)) * 100
                    
                    if rate < threshold:
                        if not state["alert_active"]:
                            triggered = True
                            details = f"Pass rate dropped to {rate:.1f}% (Threshold: {threshold}%, Window: {len(recent)})."
                    else:
                        state["alert_active"] = False # Recovered
            
            # --- DISPATCH IF TRIGGERED ---
            if triggered:
                print(f"[AlertManager] Rule '{rule.get('name')}' triggered!")
                self._dispatch_rule_alert(rule, session_id, details)
                state["alert_active"] = True

    def _dispatch_rule_alert(self, rule: dict, session_id: str, details: str):
        """Dispatches alerts for a specific rule."""
        rule_name = rule.get("name", "Unnamed Alert")
        subject = f"🚨 JerryScan AI Alert: {rule_name}"
        body = (
            f"Alert Rule triggered: {rule_name}\n"
            f"Condition Details: {details}\n\n"
            f"Latest Session: {session_id}\n\n"
            f"Please check the production line."
        )
        
        # Multiple Email Recipients
        recipients = rule.get("emails", [])
        if recipients:
            self._send_email(recipients, subject, body)
            
        # Rule-specific Webhook
        webhook_url = rule.get("webhook_url")
        if webhook_url:
            self._send_webhook(webhook_url, subject, body)

    def _send_email(self, recipients: list, subject: str, body: str):
        try:
            smtp_cfg = self.config.get("smtp", {})
            server_host = smtp_cfg.get("server")
            server_port = int(smtp_cfg.get("port", 587))
            user = smtp_cfg.get("user")
            password = smtp_cfg.get("password")

            if not all([server_host, server_port, user, password]):
                print("[AlertManager] SMTP settings incomplete.")
                return

            msg = EmailMessage()
            msg.set_content(body)
            msg['Subject'] = subject
            msg['From'] = user
            msg['To'] = ", ".join(recipients)

            if server_port == 465:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(server_host, server_port, context=context) as server:
                    server.login(user, password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(server_host, server_port) as server:
                    server.starttls()
                    server.login(user, password)
                    server.send_message(msg)
            print(f"[AlertManager] Email sent to {len(recipients)} recipients.")
        except Exception as e:
            print(f"[AlertManager] Email Error: {e}")

    def _send_webhook(self, url: str, subject: str, body: str):
        try:
            payload = {"text": f"*{subject}*\n{body}", "subject": subject, "body": body}
            requests.post(url, json=payload, timeout=5).raise_for_status()
            print("[AlertManager] Webhook dispatched.")
        except Exception as e:
            print(f"[AlertManager] Webhook Error: {e}")
