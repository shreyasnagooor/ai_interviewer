import time

class BrowserTabMonitor:
    def __init__(self):
        self.visible_tab_id = None
        self.tab_history = []
        self.suspicious_switches = 0
        self.last_check_time = None

    def set_interview_tab_id(self, tab_id):
        """Set the ID of the tab where the interview is running."""
        self.visible_tab_id = tab_id
        self.last_check_time = time.time()
        self.tab_history.append((tab_id, self.last_check_time))

    def record_tab_change(self, new_tab_id):
        """Record when the user switches tabs."""
        current_time = time.time()
        self.tab_history.append((new_tab_id, current_time))

        # Check if switching away from interview tab
        if self.visible_tab_id and new_tab_id != self.visible_tab_id:
            self.suspicious_switches += 1

        self.visible_tab_id = new_tab_id
        self.last_check_time = current_time

    def get_tab_activity(self):
        """Get statistics about tab switching behavior."""
        if len(self.tab_history) < 2:
            return {"status": "monitoring", "suspicious": False}

        total_time = self.tab_history[-1][1] - self.tab_history[0][1]
        time_on_interview = 0

        # Calculate time spent on interview tab
        for i in range(1, len(self.tab_history)):
            if self.tab_history[i-1][0] == self.visible_tab_id:
                duration = self.tab_history[i][1] - self.tab_history[i-1][1]
                time_on_interview += duration

        interview_percentage = (time_on_interview / total_time) * 100 if total_time > 0 else 100

        return {
            "status": "active",
            "time_on_interview_percentage": interview_percentage,
            "tab_switches": len(self.tab_history) - 1,
            "suspicious_switches": self.suspicious_switches,
            "is_suspicious": interview_percentage < 90 or self.suspicious_switches > 3
        }
