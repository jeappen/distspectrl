import unittest
import numpy as np
import gym.spaces

class TestEngine(unittest.TestCase):
    # Tests centralized SPECTRL monitors
    def setUp(self) -> None:
        self.horizon = 500
        
        num_agents = 2
        env = "navenv_inlineDS_planning"
        spec_id = 5

        # import Env with reward machine
        from src.main.reward_shape_PPO_wrapper import env_creator
        custom_env = lambda config: env_creator(config)
        env_config = {}
        # Choose spec from list of specs in env_creator
        env_config['spec_id'] = spec_id
        # Set to None for no reward shaping
        env_config['reward_shape_mode'] = 'sparse'
        # For setting predicate bounds
        env_config['err_arg'] = 0
        # For setting maximum distance mode in predicate
        env_config['max_mode'] = True
        # For setting num_agents
        env_config['num_agents'] = num_agents
        # For setting time horizon
        env_config['horizon'] = self.horizon

        env_config['distributed_mode'] = True
        env_config['inline_reward'] = True

        # setup if needed for future tests
        # self.env = custom_env(env_config)
        self.env = None
        self.custom_env = custom_env

        return super().setUp()
    
    def fill_env_config(self, env, spec_id, num_agents):
        env_config = {}
        # Choose spec from list of specs in env_creator
        env_config['spec_id'] = spec_id
        # Set to None for no reward shaping
        env_config['reward_shape_mode'] = 'sparse'
        # For setting predicate bounds
        env_config['err_arg'] = 0
        # For setting maximum distance mode in predicate
        env_config['max_mode'] = True
        # For setting num_agents
        env_config['num_agents'] = num_agents
        # For setting time horizon
        env_config['horizon'] = self.horizon

        env_config['spectrl_single_agent'] = 'spectrl_sa' in env
        env_config['distributed_mode'] = not env_config['spectrl_single_agent']
        env_config['inline_reward'] = env_config['distributed_mode']
        
        return env_config
    
    def step_env(self, s, goal, env, Ke_matrix, dummy_ind=1):
            # ::2 to skip goal reference dim
            error = np.dot(Ke_matrix, (goal - s[:env.spec.state_dim].reshape(-1, goal.shape[-1])[::2])).flatten()
            sys_actions = error
            
            # pick transition
            extra_action = np.zeros(env.spec.extra_action_dim)
            extra_action[dummy_ind] = 1
            
            actions = np.hstack([sys_actions, extra_action])
            s, rew, done, i = env.step(actions)
            return s, rew, done
        
    def multi_step(self, num_steps, s, goal, env, Ke_matrix, dummy_ind=1):
        for _ in range(num_steps):
            s, rew, done = self.step_env(s,goal, env, Ke_matrix, dummy_ind)
        return s, rew, done
    
    def test_centralized_mspec0(self):
        # local specs work properly when centralized mode
        env = "navenv_spectrl_sa"
        spec_id = 5
        num_agents = 2

        env_config = self.fill_env_config(env, spec_id, num_agents)
        single_env = self.custom_env(env_config)
        r = single_env.reset()
        goal = np.array([5,0])
        s = r
        Ke_matrix = np.diag([5,10])

        s, rew, done = self.multi_step(2, s, goal, single_env, Ke_matrix, 2)
        # Now only agent 1 (not agent 0) should have reached first local goal
        self.assertEqual(s[-1], 2)

        s, rew, done = self.multi_step(2, s, goal, single_env, Ke_matrix, 1)
        # Now both agents should have reached first local goal
        self.assertEqual(s[-1], 3)

        goal = np.array([0,0])
        s, rew, done = self.multi_step(4, s, goal, single_env, Ke_matrix, 1)
        
        # Now both agents should have reached end
        self.assertEqual(s[-1], 5)

    def test_centralized_mspec1(self):
        # local specs work properly when centralized mode
        env = "navenv_spectrl_sa"
        spec_id = 3
        num_agents = 2

        env_config = self.fill_env_config(env, spec_id, num_agents)
        single_env = self.custom_env(env_config)
        r = single_env.reset()
        goal = np.array([5,0])
        s = r
        Ke_matrix = np.diag([5,10])

        s, rew, done = self.multi_step(2, s, goal, single_env, Ke_matrix, 2)
        # Now only agent 1 (not agent 0) should have reached first local goal
        self.assertEqual(s[-1], 2)

        goal = np.array([[5,0], [0,0]])

        s, rew, done = self.multi_step(4, s, goal, single_env, Ke_matrix, 1)
        # Now both agents should have reached first local goal
        self.assertEqual(s[-1], 4)

        goal = np.array([0,0])
        s, rew, done = self.multi_step(6, s, goal, single_env, Ke_matrix, 1)
        
        # Now both agents should have reached end
        self.assertEqual(s[-1], 6)

        goal = np.array([3,0])
        s, rew, done = self.multi_step(4, s, goal, single_env, Ke_matrix, 1)
        
        # Now both agents should have reached end
        self.assertEqual(s[-1], 7)

    def test_centralized_mspec1_n10(self):
        # global solution to mspec1 exists when n=10
        env = "navenv_spectrl_sa"
        spec_id = 3
        num_agents = 10

        env_config = self.fill_env_config(env, spec_id, num_agents)
        single_env = self.custom_env(env_config)
        r = single_env.reset()
        s = r
        Ke_matrix = 10.0

        goal = np.array([5,0])
        s, rew, done = self.multi_step(20, s, goal, single_env, Ke_matrix, 1)

        goal = np.array([[0,0]])
        s, rew, done = self.multi_step(20, s, goal, single_env, Ke_matrix, 1)
    
        goal = np.array([3,0])
        s, rew, done = self.multi_step(20, s, goal, single_env, Ke_matrix, 1)
        
        # Now agents should have reached end
        final_monitor_state = [i for i,r in enumerate(single_env.spec.monitor.rewards) if r is not None ][0]
        self.assertEqual(s[-1], final_monitor_state)