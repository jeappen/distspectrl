import numpy as np


# Convert a system-resource predicate to a monitor predicate, i.e., over S*V
def monitor_predicate(predicate, state_dim):
    def new_predicate(state, reg):
        pvalue = predicate(state[:state_dim], state[state_dim:])
        #TODO: delete_dbg print("\nMP_1level pvalue {} state_dim {}".format(pvalue,state_dim))
        return pvalue > 0, pvalue
    return new_predicate


# True Predicate with quantitative semantics determined by the local reward bound
def true_predicate(local_reward_bound):
    def predicate(state, reg):
        return (True, local_reward_bound/2)
    return predicate


# Project predicate to a sub list of register values
# s_reg: starting register number
# e_reg: ending register number
def project_predicate(mpred, s_reg, e_reg):
    def predicate(state, reg):
        #TODO: delete_dbg print("\nPP_1level {} {}".format(s_reg,e_reg))
        return mpred(state, reg[s_reg:e_reg])
    return predicate


# Project update to a sub list of registers
def project_update(mupdate, s_reg, e_reg, clean=False):
    def update(state, reg):
        retval = reg.copy()
        if clean:
            retval = np.zeros(len(reg))
        retval[s_reg:e_reg] = mupdate(state, reg[s_reg:e_reg])
        return retval
    return update


# Project reward to a sub list of registers
def project_reward(mrew, s_reg, e_reg):
    def rew(state, reg):
        return mrew(state, reg[s_reg:e_reg])
    return rew

# Minimum of two rewards (used for BOTH)
def min_reward(mrew, mrew2):
    def rew(state, reg):
        return min(mrew(state, reg),mrew2(state, reg))
    return rew


# Reward for alw construction
def alw_reward(mrew):
    def rew(state, reg):
        return min(mrew(state, reg[:-1]), reg[-1])
    return rew


# Reward for seq construction
def seq_reward(mrew, reg_no):
    def rew(state, reg):
        rew_val = mrew(state, reg[:reg_no])
        #TODO: delete_dbg print("\n Seq rew, regno {} reg {} new_mrew {}".format(reg_no, reg, rew_val))
        return min(rew_val, reg[-1])
    return rew


# Combine predicate with reward positivity condition
# reg_init: initial register for the second monitor
def rew_pred(mpred, mrew, reg_init, s_reg, e_reg):
    def predicate(state, reg):
        (rb, rv) = mpred(state, reg_init)
        rew = mrew(state, reg[s_reg:e_reg])
        #TODO: delete_dbg print("\n rew_pred rew {} rb {} rv {}, state{} , reg_lim {}, s_reg {}, e_reg {}"
        #TODO: delete_dbg        .format(rew,rb, rv, state, reg[s_reg:e_reg], s_reg,e_reg))
        return (rb and rew > 0, min(rv, rew))
    return predicate


# Conjunction with base predicate
def conj_pred(mpred1, mpred2, reg_init):
    def predicate(state, reg):
        (b1, v1) = mpred1(state, reg_init)
        (b2, v2) = mpred2(state, reg)
        return (b1 and b2, min(v1, v2))
    return predicate


# Negation of a predicate
def neg_pred(mpred):
    def predicate(state, reg):
        (b, v) = mpred(state, reg)
        return (not b, -v)
    return predicate


# Update based on initial register value
def init_update(mupdate, reg_init):
    def update(state, reg):
        retval = reg.copy()
        retval[:len(reg_init)] = mupdate(state, reg_init)
        return retval
    return update


# Change update to track satisfaction of safety constraints
def alw_update(mupdate, mpred):
    def update(state, reg):
        return np.concatenate([mupdate(state, reg[:-1]), np.array([min(reg[-1],
                                                                       mpred(state, reg)[1])])])
    return update


# Update function for sequence
def seq_update(total_reg_no, mon1_reg_no, mon2_reg_no, mon2_init_reg, mon1_rew, mupdate):
    def update(state, reg):
        retval = np.zeros(total_reg_no)
        retval[:mon2_reg_no] = mupdate(state, mon2_init_reg)
        retval[-1] = mon1_rew(state, reg[:mon1_reg_no])
        return retval
    return update


# Identity update
def id_update(state, reg):
    return reg.copy()


# Optimistic Merges for centralized specs self loops ONLY
# Otherwise will have one transition for each agent's register update
def optimistic_predicate_merge(*args):
    # Akin to an OR
    def predicate(state, reg):
        outputs = [a(state,reg) for a in args]
        return any([a[0] for a in outputs]), max([a[1] for a in outputs])
    return predicate

def optimistic_update_merge(*args):
    # Akin to an OR update
    def update(state, reg):
        outputs = [a(state,reg) for a in args]
        return np.max(outputs, axis=0)
    return update
