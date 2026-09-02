"""Unit and component tests for the shared autonomous runtime.

Every test here drives one ``AutonomousRuntime`` directly, over an injected
stream, an in-memory session, a mounted transcript, and a recording host. The
app is deliberately absent: what is proved here is the loop, not the TUI.
"""
