import sys
import unittest
from unittest.mock import patch

from tech.tech import TECH, DandelionVoting, read_impact_hour_data
import tech.config_bounds as config_bounds


class TestDandelionVoting(unittest.TestCase):
    def setUp(self):
        # Test only the hatch setup.
        self.config = config_bounds.hatch['dandelion_voting']
        self.dandelion = DandelionVoting(total_tokens=17e6,
                                         config=self.config)

    def test_initialization(self):
        self.assertEqual(
            self.dandelion.param.support_required_percentage.bounds,
            self.config['support_required_percentage']['bounds'])
        self.assertEqual(
            self.dandelion.param.support_required_percentage.step,
            self.config['support_required_percentage']['step'])
        self.assertEqual(
            self.dandelion.support_required_percentage,
            self.config['support_required_percentage']['value'])
        self.assertEqual(
            self.dandelion.param.minimum_accepted_quorum_percentage.bounds,
            self.config['minimum_accepted_quorum_percentage']['bounds'])
        self.assertEqual(
            self.dandelion.param.minimum_accepted_quorum_percentage.step,
            self.config['minimum_accepted_quorum_percentage']['step'])
        self.assertEqual(
            self.dandelion.minimum_accepted_quorum_percentage,
            self.config['minimum_accepted_quorum_percentage']['value'])
        self.assertEqual(
            self.dandelion.vote_duration_days,
            self.config['vote_duration_days']['value'])
        self.assertEqual(
            self.dandelion.vote_buffer_hours,
            self.config['vote_buffer_hours']['value'])
        self.assertEqual(
            self.dandelion.rage_quit_hours,
            self.config['rage_quit_hours']['value'])
        self.assertEqual(
            self.dandelion.tollgate_fee_xdai,
            self.config['tollgate_fee_xdai']['value'])

    def test_support_required(self):
        self.assertEqual(
            self.dandelion.support_required(),
            self.dandelion.support_required_percentage / 100)

    def test_minimum_accepted_quorum(self):
        self.assertEqual(
            self.dandelion.minimum_accepted_quorum(),
            self.dandelion.minimum_accepted_quorum_percentage / 100)


class TestTECH(unittest.TestCase):
    def setUp(self):
        # Test only the hatch setup.
        self.impact_hour_data = read_impact_hour_data()
        self.total_impact_hours = self.impact_hour_data['Assumed IH'].sum(),
        self.config = config_bounds.hatch['tech']
        self.tech = TECH(
            total_impact_hours=self.total_impact_hours,
            impact_hour_data=self.impact_hour_data, total_cstk_tokens=1000000,
            config=self.config)

    def test_initialization(self):
        self.assertEqual(
            self.tech.total_impact_hours,
            self.total_impact_hours)
        self.assertEqual(
            self.tech.output_scenario_raise,
            self.config["output_scenario_raise"])
        self.assertEqual(
            self.tech.min_raise,
            self.config['min_max_raise']['value'][0])
        self.assertEqual(
            self.tech.param.min_raise.step,
            self.config['min_max_raise']['step'])
        self.assertEqual(
            self.tech.max_raise,
            self.config['min_max_raise']['value'][1])
        self.assertEqual(
            self.tech.param.max_raise.step,
            self.config['min_max_raise']['step'])
        self.assertEqual(
            self.tech.target_raise,
            self.config['target_raise']['value'])
        self.assertEqual(
            self.tech.param.target_raise.step,
            self.config['target_raise']['step'])
        self.assertEqual(
            self.tech.impact_hour_rate_at_target_goal,
            self.config['impact_hour_rate_at_target_goal']['value'])
        self.assertEqual(
            self.tech.param.maximum_impact_hour_rate.bounds,
            self.config['maximum_impact_hour_rate']['bounds'])
        self.assertEqual(
            self.tech.param.maximum_impact_hour_rate.step,
            self.config['maximum_impact_hour_rate']['step'])
        self.assertEqual(
            self.tech.maximum_impact_hour_rate,
            self.config['maximum_impact_hour_rate']['value'])
        self.assertEqual(
            self.tech.hatch_oracle_ratio,
            self.config['hatch_oracle_ratio']['value'])
        self.assertEqual(
            self.tech.hatch_period_days,
            self.config['hatch_period_days']['value'])
        self.assertEqual(
            self.tech.hatch_exchange_rate,
            self.config['hatch_exchange_rate']['value'])
        self.assertEqual(
            self.tech.hatch_tribute_percentage,
            self.config['hatch_tribute_percentage']['value'])

    def test_bounds_impact_hour_rate_at_target_goal(self):
        # Test if the impact_hour_rate_at_target_goal goes to 1 if it's
        # negative.
        self.tech.impact_hour_rate_at_target_goal = -1
        self.tech.bounds_impact_hour_rate_at_target_goal()
        self.assertEqual(self.tech.impact_hour_rate_at_target_goal, 1)

        # Test if the impact_hour_rate_at_target_goal goes to 1 if it's zero
        self.tech.impact_hour_rate_at_target_goal = 0
        self.tech.bounds_impact_hour_rate_at_target_goal()
        self.assertEqual(self.tech.impact_hour_rate_at_target_goal, 1)

        # Test if the impact_hour_rate_at_target_goal is higher than
        # maximum_impact_hour_rate, the impact_hour_rate_at_target_goal will be
        # bounded by the maximum_impact_hour_rate.
        self.tech.impact_hour_rate_at_target_goal = 500
        self.tech.maximum_impact_hour_rate = 400
        self.tech.bounds_impact_hour_rate_at_target_goal()
        self.assertEqual(self.tech.impact_hour_rate_at_target_goal,
                         self.tech.maximum_impact_hour_rate)

    def test_get_impact_hour_slope(self):
        # Test the equation from impact_hour_rate_at_target_goal to
        # impact_hour_rate.
        self.tech.maximum_impact_hour_rate = 500
        self.tech.impact_hour_rate_at_target_goal = 150
        self.tech.target_rais = 2500000
        self.tech.total_impact_hours = 9730.726999999999
        impact_hour_slope = self.tech.get_impact_hour_slope()
        self.assertEqual(impact_hour_slope, 599.4755924540206)

    def test_bounds_target_raise(self):
        self.tech.min_raise = 100_000
        self.tech.max_raise = 50_000_000

        # Test if target_raise is higher than max_raise
        self.tech.target_raise = 51_000_000
        self.tech.bounds_target_raise()
        self.assertEqual(self.tech.target_raise, self.tech.max_raise)

        # Test if target_raise is lower than min_raise
        self.tech.target_raise = 99_000
        self.tech.bounds_target_raise()
        self.assertEqual(self.tech.target_raise, self.tech.min_raise)


if __name__ == '__main__':
    unittest.main()
