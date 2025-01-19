def linear_schedule(initial_value):
    def schedule(progress_remaining):
        return progress_remaining * initial_value  # Linearly decrease
    return schedule
