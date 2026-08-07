"""Optional local email/webhook alert rules."""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage

import requests

from .config import ConfigManager


class AlertManager:
    def __init__(self, config_manager: ConfigManager, history_manager=None):
        self.config = config_manager
        self.history_manager = history_manager
        self.rule_states: dict[str, dict[str, object]] = {}
        self.timeout = float(os.getenv("JERRYSCAN_ALERT_TIMEOUT_SECONDS", "5"))

    def _state(self, rule_id: str) -> dict[str, object]:
        return self.rule_states.setdefault(
            rule_id, {"streak": 0, "alert_active": False}
        )

    def evaluate_session(self, overall_status: str, session_id: str) -> None:
        # WRONG_INPUT is intentionally ignored: it is not a defective can and
        # must not affect failure streaks or pass-rate alerts.
        for rule in self.config.get("alerts", []):
            if not rule.get("enabled", True):
                continue
            state = self._state(str(rule.get("id", "unnamed")))
            triggered = False
            details = ""
            if rule.get("type") == "consecutive_fails":
                if overall_status == "FAIL":
                    state["streak"] = int(state["streak"]) + 1
                    threshold = int(rule.get("threshold", 3))
                    if int(state["streak"]) >= threshold and not state["alert_active"]:
                        triggered = True
                        details = f"{state['streak']} consecutive failures detected."
                elif overall_status == "PASS":
                    state.update(streak=0, alert_active=False)
            elif rule.get("type") == "pass_rate" and self.history_manager:
                window = int(rule.get("window", 50))
                threshold = float(rule.get("threshold", 90))
                recent = [
                    session
                    for session in self.history_manager.get_history(limit=max(window * 3, window))
                    if session.get("overall_status") in {"PASS", "FAIL"}
                ][:window]
                if len(recent) >= 5:
                    passes = sum(item["overall_status"] == "PASS" for item in recent)
                    rate = passes / len(recent) * 100
                    if rate < threshold and not state["alert_active"]:
                        triggered = True
                        details = f"Pass rate is {rate:.1f}% across {len(recent)} decisions."
                    elif rate >= threshold:
                        state["alert_active"] = False
            if triggered:
                self._dispatch(rule, session_id, details)
                state["alert_active"] = True

    def _dispatch(self, rule: dict, session_id: str, details: str) -> None:
        name = rule.get("name", "Unnamed Alert")
        subject = f"JerryScan AI Alert: {name}"
        body = (
            f"Alert rule: {name}\n"
            f"Details: {details}\n\n"
            f"Latest session: {session_id}\n"
        )
        recipients = rule.get("emails", [])
        if recipients:
            self._send_email(recipients, subject, body)
        if rule.get("webhook_url"):
            self._send_webhook(rule["webhook_url"], subject, body)

    def _send_email(self, recipients: list[str], subject: str, body: str) -> None:
        try:
            smtp = self.config.get("smtp", {})
            host = smtp.get("server")
            port = int(smtp.get("port", 587))
            user = smtp.get("user")
            password = smtp.get("password")
            if not all((host, port, user, password)):
                print("[AlertManager] SMTP settings are incomplete.")
                return
            message = EmailMessage()
            message.set_content(body)
            message["Subject"] = subject
            message["From"] = user
            message["To"] = ", ".join(recipients)
            context = ssl.create_default_context()
            if port == 465:
                with smtplib.SMTP_SSL(
                    host, port, context=context, timeout=self.timeout
                ) as server:
                    server.login(user, password)
                    server.send_message(message)
            else:
                with smtplib.SMTP(host, port, timeout=self.timeout) as server:
                    server.starttls(context=context)
                    server.login(user, password)
                    server.send_message(message)
        except Exception as exc:
            print(f"[AlertManager] Email failed: {type(exc).__name__}: {exc}")

    def _send_webhook(self, url: str, subject: str, body: str) -> None:
        try:
            requests.post(
                url,
                json={"text": f"*{subject}*\n{body}", "subject": subject, "body": body},
                timeout=self.timeout,
            ).raise_for_status()
        except Exception as exc:
            print(f"[AlertManager] Webhook failed: {type(exc).__name__}: {exc}")
