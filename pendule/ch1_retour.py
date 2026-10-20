""" 
Pour PPO (et REINFORCE), le cycle est :

- jouer des épisodes avec la politique actuelle ;
- apprendre de ces épisodes (corriger la politique) ;
- rejouer de nouveaux épisodes avec la politique corrigée, et ainsi de suite.
"""

import numpy as np
from pendule_env import Pendule


def jouer_episode(env, politique):
    #Joue un épisode, composé de 200 pas.
    """Fait jouer la politique jusqu'à la fin de l'épisode et garde tout."""
    observation, fini = env.reset(), False
    observations, actions, rewards = [], [], []
    while not fini:
        u = politique(observation)
        observations.append(observation)
        actions.append(u)

        # Reward est la récompense obtenue à chaque pas. Elle vaut l'opposé du coût. Le cout valant lui même th_norm**2 + 0.1 * thdot**2 + 0.001 * u**2
        observation, reward, fini = env.step(u)
        rewards.append(reward)
    return np.array(observations), np.array(actions), np.array(rewards)


def discounted_return(rewards, gamma):
    """G[t] = r[t] + gamma * r[t+1] + gamma^2 * r[t+2] + ...  (calculé à l'envers)"""
    G = np.zeros_like(rewards)        # zeros_like crée un tableau de la même forme que rewards, mais avec des zéros.
    suite = 0.0                       # "ce qui vient après" le dernier pas = 0

    # On parcourt à l'envers car ça économise des calculs.
    for t in reversed(range(len(rewards))):
        suite = rewards[t] + gamma * suite
        G[t] = suite
    return G


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    env = Pendule(seed=0)

    # Politique aléatoire : on choisit une action aléatoire entre -2 et 2.
    hasard = lambda obs: rng.uniform(-2, 2) 

    observations, actions, rewards = jouer_episode(env, hasard)
    print("nombre de pas :", len(rewards))
    print("récompense au pas 0, 50, 100, 199 :", rewards[[0, 50, 100, 199]].round(2))
    print("somme des récompenses :", rewards.sum().round(1))


    # Un gamma faible met peu en avant les récompenses à long terme.
    # Au contraire, un gamma élevé met beaucoup en avant les récompenses à long terme.
    # Et on le voit grâce a la formule du discounted return : G[t] = r[t] + gamma * r[t+1] + gamma^2 * r[t+2] + ...

    for gamma in (0.0, 0.9, 0.95, 0.99, 1.0):    # Même épisode, vu avec différents gamma
        G = discounted_return(rewards, gamma)
        print(f"gamma = {gamma:4} | G[0] = {G[0]:8.1f} | G[100] = {G[100]:8.1f} | G[199] = {G[199]:6.2f}")

    #Ici, on ne joue qu'un seul épisode, s'il y en avait plusieurs, on aurait J, la note de lapolitique, qui est la moyenne de G₀ sur beaucoup d'épisodes.

    # Vérification : la récursion donne bien la somme explicite
    gamma = 0.95
    explicite = sum(gamma ** k * rewards[k] for k in range(len(rewards)))
    print("G[0] récursif :", discounted_return(r, gamma)[0].round(4), "| somme explicite :", round(explicite, 4))