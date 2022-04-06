from environs import Env


env = Env(expand_vars=True)
env.read_env()
