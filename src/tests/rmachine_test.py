import unittest
import numpy as np
import gym.spaces

class TestEngine(unittest.TestCase):
    def setUp(self) -> None:
        num_agents = 2
        horizon = 500
        env = "navenv_inlineDS_planning"
        spec_id = 5

        # import Env with reward machine
        from rm_cooperative_marl.src.experiments.ray_rm_env import rm_env_creator,CustomCallbacks
        from src.main.reward_shape_PPO_wrapper import custom_eval_function
        custom_env = lambda config: rm_env_creator(config)
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
        env_config['num_agents'] = 3
        # For setting time horizon
        env_config['horizon'] = horizon

        env_config['distributed_mode'] = True
        env_config['inline_reward'] = True
        # TODO: Set eval mode to not use thres in labelling and do actual MA env
        eval_env_config = env_config.copy()
        eval_env_config['marl_rm_eval_mode'] = True
        eval_fn = custom_eval_function


        # TODO: REMOVE
        # env_config['marl_rm_eval_mode'] = True

        single_env = custom_env(env_config)
        self.env = single_env

        return super().setUp()

    def test_rmachine_reaches_goal(self):
        """ Test reaching the rendezvous point then goal.
        """
        r = self.env.reset()
        final_rm_u = max(self.env.agent_list['agent-0'].rm.delta_u)
        goal = np.array([5,0])
        # goal = np.array([0,0])
        s = r
        Ke = 5
        total_reward = {a:0 for a in r}
        for _ in range(10):
            # Simple PID control
            actions = {a: Ke*(goal-s[a][:goal.shape[0]]) for a in s}
            s,rew,done,i = self.env.step(actions)
            for a in rew:
                total_reward[a] += rew[a]
        
        rew_stage1 = total_reward.copy()
        goal = np.array([0,0])
        for _ in range(10):
            # Simple PID control
            actions = {a: Ke*(goal-s[a][:goal.shape[0]]) for a in s}
            s,rew,done,i = self.env.step(actions)
            for a in rew:
                total_reward[a] += rew[a]
        
        rew_stage2 = total_reward.copy()

        for a in s:
            print("\n tc1: {} rw2 {} rw1 {}".format(a,rew_stage2[a], rew_stage1[a]))
            self.assertGreater(rew_stage2[a], rew_stage1[a])
            self.assertEqual(s[a][-1],final_rm_u) # Final rmachine state

    def test_rmachine_doesnt_loiter(self):
        # Test hanging around rdvz 1 giving worse reward than shifting
        r = self.env.reset()
        final_rm_u = max(self.env.agent_list['agent-0'].rm.delta_u)
        goal = np.array([5,0])
        # goal = np.array([0,0])
        s = r
        Ke = 5
        total_reward = {a:0 for a in r}
        for _ in range(10):
            # Simple PID control
            actions = {a: Ke*(goal-s[a][:goal.shape[0]]) for a in s}
            s,rew,done,i = self.env.step(actions)
            for a in rew:
                total_reward[a] += rew[a]
        
        rew_stage1 = total_reward.copy()
        goal = np.array([3,0]) # exiting rdvz1 region
        for _ in range(10):
            # Simple PID control
            actions = {a: Ke*(goal-s[a][:goal.shape[0]]) for a in s}
            s,rew,done,i = self.env.step(actions)
            for a in rew:
                total_reward[a] += rew[a]
        
        rew_stage2 = total_reward.copy()

        for a in s:
            print("\n tc2: {} rw2 {} rw1 {}".format(a,rew_stage2[a], rew_stage1[a]))
            self.assertGreater(rew_stage1[a], rew_stage2[a])
            # self.assertEqual(s[a][-1],final_rm_u) # Final rmachine state


    def test_rmachine_doesnt_hang_near_rdvz1(self):
        # Test hanging just outside rdvz 1 giving worse reward than shifting towards goal
        r = self.env.reset()
        final_rm_u = max(self.env.agent_list['agent-0'].rm.delta_u)
        goal = np.array([4,0])
        # goal = np.array([0,0])
        s = r
        Ke = 5
        total_reward = {a:0 for a in r}
        for _ in range(10):
            # Simple PID control
            actions = {a: Ke*(goal-s[a][:goal.shape[0]]) for a in s}
            s,rew,done,i = self.env.step(actions)
            for a in rew:
                total_reward[a] += rew[a]
        
        rew_stage1 = total_reward.copy()

        goal = np.array([5,0])
        for _ in range(10):
            # Simple PID control
            actions = {a: Ke*(goal-s[a][:goal.shape[0]]) for a in s}
            s,rew,done,i = self.env.step(actions)
            for a in rew:
                total_reward[a] += rew[a]
        
        goal = np.array([0,0])

        for _ in range(10):
            # Simple PID control
            actions = {a: Ke*(goal-s[a][:goal.shape[0]]) for a in s}
            s,rew,done,i = self.env.step(actions)
            for a in rew:
                total_reward[a] += rew[a]
        
        rew_stage2 = total_reward.copy()

        for a in s:
            print("\n tc3: {} rw2 {} rw1 {}".format(a,rew_stage2[a], rew_stage1[a]))
            self.assertGreater(rew_stage2[a], rew_stage1[a])
            # self.assertEqual(s[a][-1],final_rm_u) # Final rmachine state
