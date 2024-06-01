cache = dict()

class RuntimeReward:
    @staticmethod
    def estimate_running_time(env, use_cache=True):
        prgm = env.observation["Ir"]
        if not use_cache or not prgm in cache:
            try:
                cache[prgm] = env.observation["Runtime"].mean().item()
            except Exception as e:
                print("execution of prgm failed")
                print(e)
                cache[prgm] = 300*60
        return cache[prgm]

    def __init__(self, initial_env):
        self.reference_time = RuntimeReward.estimate_running_time(initial_env)

    def rel_reward(self, env, running_time=None):
        if running_time is None:
            running_time = RuntimeReward.estimate_running_time(env)
        delta = self.reference_time - running_time
        self.reference_time = running_time
        return delta
