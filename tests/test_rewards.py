import unittest

import jax
import jax.numpy as jnp
from ksim import Trajectory

from ksim_kbot.rewards import FeetAirTimeReward


class TestFeetAirTimeReward(unittest.TestCase):
    """Test class for FeetAirTimeReward to verify air time computation logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.reward = FeetAirTimeReward(scale=1.0, threshold_min=0.0, threshold_max=0.4, ctrl_dt=0.02)
        self.rng = jax.random.PRNGKey(0)

    def test_initial_carry(self):
        """Test that initial_carry returns the expected structure."""
        initial_carry = self.reward.initial_carry(self.rng)

        # Check that the carry has the expected keys
        self.assertIn("first_contact", initial_carry)
        self.assertIn("last_contact", initial_carry)
        self.assertIn("feet_air_time", initial_carry)

        # Check that the values have the expected shapes and types
        self.assertEqual(initial_carry["first_contact"].shape, (2,))
        self.assertEqual(initial_carry["last_contact"].shape, (2,))
        self.assertEqual(initial_carry["feet_air_time"].shape, (2,))

        # Check that the values are initialized to zeros
        self.assertTrue(jnp.all(initial_carry["first_contact"] == False))
        self.assertTrue(jnp.all(initial_carry["last_contact"] == False))
        self.assertTrue(jnp.all(initial_carry["feet_air_time"] == 0.0))

    def test_air_time_computation(self):
        """Test that air time is computed correctly for a simple trajectory."""
        # Create a simple trajectory with feet contact observations
        # Format: [timestep, foot_id]
        # 1 means foot is in contact, 0 means foot is in the air
        contact_sequence = jnp.array(
            [
                [0, 0],  # Both feet in air (start in air)
                [0, 0],  # Both feet in air
                [0, 0],  # Both feet in air
                [1, 1],  # Both feet in contact (first contact)
                [0, 0],  # Both feet in air
                [0, 0],  # Both feet in air
                [0, 0],  # Both feet in air
                [1, 1],  # Both feet in contact
                [0, 0],  # Both feet in air
            ]
        )

        # Create a trajectory object with the contact sequence
        trajectory = Trajectory(
            qpos=jnp.zeros((9, 10)),  # Dummy qpos
            qvel=jnp.zeros((9, 10)),  # Dummy qvel
            xpos=jnp.zeros((9, 10, 3)),  # Dummy xpos
            xquat=jnp.zeros((9, 10, 4)),  # Dummy xquat
            obs={"feet_contact_observation": contact_sequence},
            command={},  # Dummy command
            event_state={},  # Dummy event_state
            action=jnp.zeros((9, 10)),  # Dummy action
            done=jnp.array([False, False, False, False, False, False, False, False, True]),  # Last step is done
            success=jnp.array([False, False, False, False, False, False, False, False, True]),  # Last step is success
            timestep=jnp.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]),  # Timesteps
            termination_components={},  # Dummy termination_components
            aux_outputs={},  # Dummy aux_outputs
        )

        # Get initial carry
        initial_carry = self.reward.initial_carry(self.rng)

        # Compute rewards
        rewards, final_carry = self.reward(trajectory, initial_carry)

        # Print debug information
        print(f"Rewards: {rewards}")
        print(f"Final carry: {final_carry}")

        # For the first contact (step 3):
        # Air time: 3 steps * 0.02 = 0.06 seconds
        # This is within the threshold range (0.0 to 0.4)
        # So the reward should be positive

        # For the second contact (step 7):
        # Air time: 3 steps * 0.02 = 0.06 seconds
        # This is within the threshold range (0.0 to 0.4)
        # So the reward should be positive
        breakpoint()
        # Check that the rewards are positive at the contact steps
        self.assertTrue(rewards[3] > 0, f"Expected positive reward at first contact, got {rewards[3]}")
        self.assertTrue(rewards[7] > 0, f"Expected positive reward at second contact, got {rewards[7]}")

        # Check that the rewards are zero at non-contact steps
        self.assertEqual(rewards[0], 0, f"Expected zero reward at non-contact step, got {rewards[0]}")
        self.assertEqual(rewards[1], 0, f"Expected zero reward at non-contact step, got {rewards[1]}")
        self.assertEqual(rewards[2], 0, f"Expected zero reward at non-contact step, got {rewards[2]}")
        self.assertEqual(rewards[4], 0, f"Expected zero reward at non-contact step, got {rewards[4]}")
        self.assertEqual(rewards[5], 0, f"Expected zero reward at non-contact step, got {rewards[5]}")
        self.assertEqual(rewards[6], 0, f"Expected zero reward at non-contact step, got {rewards[6]}")
        self.assertEqual(rewards[8], 0, f"Expected zero reward at non-contact step, got {rewards[8]}")

        # Check that the final carry has the expected values
        self.assertTrue(jnp.all(final_carry["first_contact"] == False))
        self.assertTrue(jnp.all(final_carry["last_contact"] == False))
        self.assertTrue(jnp.all(final_carry["feet_air_time"] == 0.0))

    def test_air_time_thresholds(self):
        """Test that air time rewards are correctly applied based on thresholds."""
        # Create a trajectory with varying air times
        # We'll test different scenarios:
        # 1. Air time below threshold_min (should give no reward)
        # 2. Air time within threshold range (should give positive reward)
        # 3. Air time above threshold_max (should give capped reward)

        # Create a trajectory with feet contact observations
        # Format: [timestep, foot_id]
        # 1 means foot is in contact, 0 means foot is in the air
        contact_sequence = jnp.array(
            [
                [0, 0],  # Both feet in air (start in air)
                [0, 0],  # Both feet in air
                [0, 0],  # Both feet in air
                [0, 0],  # Both feet in air
                [0, 0],  # Both feet in air
                [1, 1],  # Both feet in contact (first contact)
                [0, 0],  # Both feet in air
                [0, 0],  # Both feet in air
                [0, 0],  # Both feet in air
                [0, 0],  # Both feet in air
                [0, 0],  # Both feet in air
                [1, 1],  # Both feet in contact
                [0, 0],  # Both feet in air
            ]
        )

        # Create a trajectory object with the contact sequence
        trajectory = Trajectory(
            qpos=jnp.zeros((13, 10)),  # Dummy qpos
            qvel=jnp.zeros((13, 10)),  # Dummy qvel
            xpos=jnp.zeros((13, 10, 3)),  # Dummy xpos
            xquat=jnp.zeros((13, 10, 4)),  # Dummy xquat
            obs={"feet_contact_observation": contact_sequence},
            command={},  # Dummy command
            event_state={},  # Dummy event_state
            action=jnp.zeros((13, 10)),  # Dummy action
            done=jnp.array(
                [False, False, False, False, False, False, False, False, False, False, False, False, True]
            ),  # Last step is done
            success=jnp.array(
                [False, False, False, False, False, False, False, False, False, False, False, False, True]
            ),  # Last step is success
            timestep=jnp.array(
                [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24]
            ),  # Timesteps
            termination_components={},  # Dummy termination_components
            aux_outputs={},  # Dummy aux_outputs
        )

        # Get initial carry
        initial_carry = self.reward.initial_carry(self.rng)

        # Compute rewards
        rewards, final_carry = self.reward(trajectory, initial_carry)

        # Print debug information
        print(f"Default threshold rewards: {rewards}")

        # For the first contact (step 5):
        # Air time: 5 steps * 0.02 = 0.1 seconds
        # This is within the threshold range (0.0 to 0.4)
        # So the reward should be positive

        # For the second contact (step 11):
        # Air time: 5 steps * 0.02 = 0.1 seconds
        # This is within the threshold range (0.0 to 0.4)
        # So the reward should be positive

        # Check that the rewards are positive at the contact steps
        self.assertTrue(rewards[5] > 0, f"Expected positive reward at first contact, got {rewards[5]}")
        self.assertTrue(rewards[11] > 0, f"Expected positive reward at second contact, got {rewards[11]}")

        # Now let's test with a reward that has different thresholds
        reward_high_threshold = FeetAirTimeReward(
            scale=1.0,
            threshold_min=0.2,
            threshold_max=0.5,
            ctrl_dt=0.02,  # Higher minimum threshold
        )

        # Compute rewards with the new threshold
        rewards_high, _ = reward_high_threshold(trajectory, initial_carry)

        # Print debug information
        print(f"High threshold rewards: {rewards_high}")

        # With the higher threshold, the air time (0.1) is below threshold_min (0.2)
        # So the reward should be zero
        self.assertTrue(jnp.all(rewards_high == 0), f"Expected zero rewards with high threshold, got {rewards_high}")

        # Now let's test with a reward that has a very low threshold_max
        reward_low_max = FeetAirTimeReward(
            scale=1.0,
            threshold_min=0.05,
            threshold_max=0.15,
            ctrl_dt=0.02,  # Lower maximum threshold
        )

        # Compute rewards with the new threshold
        rewards_low_max, _ = reward_low_max(trajectory, initial_carry)

        # Print debug information
        print(f"Low max threshold rewards: {rewards_low_max}")

        # With the lower threshold_max, the air time (0.1) is within the range
        # So the reward should be positive
        self.assertTrue(rewards_low_max[5] > 0, f"Expected positive reward at first contact, got {rewards_low_max[5]}")
        self.assertTrue(
            rewards_low_max[11] > 0, f"Expected positive reward at second contact, got {rewards_low_max[11]}"
        )


if __name__ == "__main__":
    unittest.main()
