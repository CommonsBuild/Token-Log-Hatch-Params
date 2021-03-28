hatch = {
    "title": "TEC Hatch Dashboard",
    "url": "https://params.tecommons.org/hatch",
    "tech": {
        "min_max_raise": {
            "value": (100000, 50000000),
            "bounds": (10000, 95000000),
        },
        "target_raise": {
            "value": 2500000,
            "bounds": (10000, 95000000),
            "step": 100,
        },
        "impact_hour_slope": {
            "value": 1000,
            "bounds": (0, 10000),
            "step": 1,
        },
        "maximum_impact_hour_rate": {
            "value": 500,
            "bounds": (0, 1000),
            "step": 1,
        },
        "hatch_oracle_ratio": {
            "value": 10,
            "bounds": (0.3, 200),
            "step": 0.1,
        },
        "hatch_period_days": {
            "value": 15,
            "bounds": (5, 30),
            "step": 1,
        },
        "hatch_exchange_rate": {
            "value": 10000,
            "bounds": (1, 100000),
            "step": 10,
        },
        "hatch_tribute_percentage": {
            "value": 5,
            "bounds": (0, 100),
            "step": 1,
        },
        "output_scenario_raise": [
            300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000,
            1250000, 1500000, 1750000, 2000000, 2500000, 3000000, 3500000,
            4000000, 4500000, 5000000, 6000000, 7000000, 8000000, 9000000,
            10000000, 11000000, 12000000, 13000000, 14000000, 15000000,
            20000000, 25000000, 30000000, 40000000, 50000000, 60000000,
            70000000, 80000000, 95000000
        ],
    },
    "dandelion_voting": {
        "support_required_percentage": {
            "value": 60,
            "bounds": (50, 90),
            "step": 1,
        },
        "minimum_accepted_quorum_percentage": {
            "value": 2,
            "bounds": (0, 50),
            "step": 1,
        },
        "vote_duration_days": {
            "value": 3,
            "bounds": (1, 14),
            "step": 1,
        },
        "vote_buffer_hours": {
            "value": 8,
            "bounds": (6, 36),
            "step": 1,
        },
        "rage_quit_hours": {     # Change it to days (1-14 days)
            "value": 1,
            "bounds": (1, 14),
            "step": 1,
        },
        "tollgate_fee_xdai": {
            "value": 1000,
            "bounds": (100, 3000),
            "step": 10,
        },
    },
}

# Add all the classes parameters to the config file, all of them on a tuple
# format to represent the boundaries.
test_hatch = {
    "title": "TEC Test Hatch Dashboard",
    "url": "https://params.tecommons.org/test-hatch",
    "tech": {
        "min_max_raise": {
            "value": (5, 1000),
            "bounds": (5, 1000),
        },
        "target_raise": {
            "value": 500,
            "bounds": (5, 1000),
            "step": 1,
        },
        "impact_hour_slope": {
            "value": 0.01,
            "bounds": (0, 1),
            "step": 0.01,
        },
        "maximum_impact_hour_rate": {
            "value": 0.01,
            "bounds": (0, 1),
            "step": 0.01,
        },
        "hatch_oracle_ratio": {
            "value": 0.01,
            "bounds": (0, 1),
            "step": 0.01,
        },
        "hatch_period_days": {
            "value": 15,
            "bounds": (5, 30),
            "step": 1,
        },
        "hatch_exchange_rate": {
            "value": 10000,
            "bounds": (1, 100000),
            "step": 10,
        },
        "hatch_tribute_percentage": {
            "value": 5,
            "bounds": (0, 100),
            "step": 1,
        },
        "output_scenario_raise": [
            1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 150, 175,
            200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000
        ],
    },
    "dandelion_voting": {
        "support_required_percentage": {
            "value": 60,
            "bounds": (50, 90),
            "step": 1,
        },
        "minimum_accepted_quorum_percentage": {
            "value": 2,
            "bounds": (0, 50),
            "step": 1,
        },
        "vote_duration_days": {
            "value": 3,
            "bounds": (1, 14),
            "step": 1,
        },
        "vote_buffer_hours": {
            "value": 8,
            "bounds": (1, 48),
            "step": 1,
        },
        "rage_quit_hours": {     # Change it to days (1-14 days)
            "value": 1,
            "bounds": (1, 48),
            "step": 1,
        },
        "tollgate_fee_xdai": {
            "value": 5,
            "bounds": (1, 100),
            "step": 1,
        },
    },
}
