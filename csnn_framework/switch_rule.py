import numpy as np


class SwitchRule():
    """
    Implements an improved switching rule that prevents getting stuck with suboptimal controllers.
    """

    def __init__(self, controller_names, default_controller=None, use_stability_check=False):
        if not controller_names:
            raise ValueError("controller_names cannot be empty")

        self.controller_names = controller_names
        self.use_stability_check = use_stability_check
        self.default_controller = default_controller or controller_names[0]
        self.reset()

    def select_controller(self, j_predictions, stability_flags=None):
        """
        Improved controller selection that prevents getting stuck.
        """
        # Validate inputs
        if not all(name in j_predictions for name in self.controller_names):
            missing = set(self.controller_names) - set(j_predictions.keys())
            raise ValueError(f"Missing cost predictions for controllers: {missing}")

        # Always allow the current controller
        allowed_controllers = set([self.previous_controller_name])

        # CRITICAL FIX: Allow switching to ANY controller that's significantly better
        # than the current one, regardless of historical performance
        current_cost = j_predictions[self.previous_controller_name]

        for name in self.controller_names:
            if name != self.previous_controller_name:
                # Check availability
                is_available = True
                if self.use_stability_check and stability_flags is not None:
                    is_available = stability_flags.get(name, True)

                if is_available and not np.isnan(j_predictions[name]) and not np.isinf(j_predictions[name]):
                    # NEW LOGIC: Allow switching if either:
                    # 1. The controller has improved since last time it was used (original rule), OR
                    # 2. It's significantly better than the current controller (new rule to prevent stuckness)
                    cost_improvement = j_predictions[name] < self.L_values[name]
                    significantly_better = j_predictions[name] < current_cost * 0.8  # 20% better

                    if cost_improvement or significantly_better:
                        allowed_controllers.add(name)

        # Select the best controller from allowed set
        allowed_costs = {name: j_predictions[name] for name in allowed_controllers}

        if not allowed_costs:
            # Fallback to minimum cost
            return min(j_predictions, key=j_predictions.get)

        return min(allowed_costs, key=allowed_costs.get)

    def update_state(self, chosen_controller_name, j_predictions):
        """
        Update the internal state after a controller is selected.
        """
        if chosen_controller_name not in self.controller_names:
            raise ValueError(f"Unknown controller: {chosen_controller_name}")

        # Update L value for the active controller
        current_cost = j_predictions[chosen_controller_name]
        if not np.isnan(current_cost) and not np.isinf(current_cost):
            self.L_values[chosen_controller_name] = min(current_cost, self.L_values[chosen_controller_name])

        # Update previous controller
        self.previous_controller_name = chosen_controller_name

    def reset(self):
        """Reset the switching rule to initial state."""
        self.L_values = {name: float('inf') for name in self.controller_names}
        self.previous_controller_name = self.default_controller
        print(f"SwitchRule reset. Default controller: {self.default_controller}")

    def get_state(self):
        """Return current state for debugging."""
        return {'previous_controller': self.previous_controller_name, 'L_values': self.L_values.copy(), 'controllers': self.controller_names}


# --- Test the improved switching logic ---
def test_switching_scenario():
    """Test the scenario you described: VF -> SM -> LQR."""
    switcher = SwitchRule(["LQR", "SM", "VF"], default_controller="VF")

    # Simulate your scenario
    print("=== Testing VF -> SM -> LQR scenario ===")

    # Step 1: VF has lowest cost initially
    costs_step1 = {"LQR": 15.0, "SM": 12.0, "VF": 8.0}
    selected = switcher.select_controller(costs_step1)
    switcher.update_state(selected, costs_step1)
    print(f"Step 1 - Selected: {selected}, Costs: {costs_step1}")
    print(f"L_values: {switcher.L_values}")

    # Step 2: SM becomes better than VF, but VF's L_value is still 8.0
    # SM cost (7.0) < VF's current cost (9.0) and significantly better (7.0 < 9.0*0.8 = 7.2)
    costs_step2 = {"LQR": 14.0, "SM": 7.0, "VF": 9.0}
    selected = switcher.select_controller(costs_step2)
    switcher.update_state(selected, costs_step2)
    print(f"Step 2 - Selected: {selected}, Costs: {costs_step2}")
    print(f"L_values: {switcher.L_values}")

    # Step 3: LQR becomes best, but SM's L_value is 7.0 and VF's is 8.0
    # LQR cost (6.0) < SM's current cost (8.0) and significantly better (6.0 < 8.0*0.8 = 6.4)
    costs_step3 = {"LQR": 6.0, "SM": 8.0, "VF": 10.0}
    selected = switcher.select_controller(costs_step3)
    switcher.update_state(selected, costs_step3)
    print(f"Step 3 - Selected: {selected}, Costs: {costs_step3}")
    print(f"L_values: {switcher.L_values}")


if __name__ == "__main__":
    test_switching_scenario()
