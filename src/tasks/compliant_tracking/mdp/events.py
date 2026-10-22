"""Policy-rate force evaluation with body-local holding across physics steps."""


def cache_forcefield_wrench(env, env_ids=None, action_name="forcefield_wrench"):
    # Step events run after the command manager, matching IsaacLab's interval
    # event placement. A .005s interval does not run inside decimation there.
    del env_ids
    env.action_manager.get_term(action_name).cache_wrench()
