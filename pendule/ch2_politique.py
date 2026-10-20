##Ce qu'on apprend
# Le log de la densité d'une loi normale N(mu, sigma²) est donnée par la fonction log_prob.
# plus sigma est grand, moins la fonction aura de confiance, même quand a=mu.

"""oici une image concrète. 
Si μ = 0,2 et que tu as tiré a = 1,0, alors (a − μ)/σ² est positif : augmenter μ rend 1,0 moins surprenant. 
Si l’action 1,0 s’est révélée bonne, on veut qu’elle redevienne plus probable, donc on augmente μ. 
Si elle s’est révélée mauvaise, on fait l’inverse. 
Tout REINFORCE tient dans cette phrase."""

##Vocabulaire
# a : action
# mu : moyenne
# sigma : écart-type
# w : poids
# obs : observation
# fini : booléen indiquant si l'épisode est fini
# O : liste des observations
# A : liste des actions
# LP : liste des log_probs

import numpy as np
from pendule_env import Pendule

LOG_2PI = np.log(2 * np.pi)


def log_prob(a, mu, sigma):
    """log de la densité d'une loi normale N(mu, sigma²) en a."""

    # Dit autrement, cette fonction permet de calculer la probabilité d'une action a, sachant la moyenne mu et l'écart-type sigma.
    # A noter que la fonction est maximum quand a = mu, et décroit symétriquement de part et d'autre de mu.
    
    # A noter également que la fonction peut sortir des valeurs poisitives comme négatives, l'importance réside dans les différences plus que la valeur.
    return -0.5 * ((a - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * LOG_2PI


class PolitiqueGaussienne:
    """mu dépend de l'observation (ici : une combinaison linéaire), sigma est un réglage."""

    def __init__(self, w, sigma, seed=0):
        self.w, self.sigma = np.array(w, dtype=float), sigma
        self.rng = np.random.default_rng(seed)

    def mu(self, obs):
        # calcul mathématique mu = w1 * cos(th) + w2 * sin(th) + w3 * vitesse
        # chaque élément observée a un poids w
        return float(obs @ self.w)

    def agir(self, obs):
        mu = self.mu(obs)
        a = mu + self.sigma * self.rng.standard_normal()     # tirage dans N(mu, sigma²)
        # Un tirage loin de mu ne signifie pas que l'action est mauvaise, elle peut très bien se révélée positive, et l'objectif sera alors de rapporcher mu de a.
        
        # retourne l'action ainsi que la densité de probabilité de l'action.
        return a, log_prob(a, mu, self.sigma)


if __name__ == "__main__":
    # --- 1) La loi de probabilité autour d'une moyenne -----------------------
    print("1) log_prob de différentes actions, pour mu = 0.5 et sigma = 1 :")
    for a in (-1.5, 0.0, 0.5, 1.0, 2.5):
        print(f"   a = {a:+.1f} | log_prob = {log_prob(a, 0.5, 1.0):+.3f}")

    print("\n2) Effet de sigma sur la probabilité de l'action la plus probable (a = mu) :")
    for sigma in (2.0, 1.0, 0.3, 0.1):
        print(f"   sigma = {sigma:3} | log_prob(a = mu) = {log_prob(0.0, 0.0, sigma):+.3f}")

    # --- 2) Le gradient de log_prob par rapport à mu ------------------------
    a, mu, sigma, h = 1.0, 0.2, 0.8, 1e-6
    # Gradient de log_prob par rapport a mu: dlog_prob/dmu = (a - mu) / sigma ** 2
    analytique = (a - mu) / sigma ** 2
    numerique = (log_prob(a, mu + h, sigma) - log_prob(a, mu - h, sigma)) / (2 * h)
    print(f"\n3) d log_prob / d mu : formule = {analytique:.5f} | différence finie = {numerique:.5f}")
    print("   -> positif quand a > mu : augmenter mu rapproche mu de a, donc rend a plus probable.")

    # --- 3) Un épisode joué par une politique stochastique ------------------
    env = Pendule(seed=0)
    politique = PolitiqueGaussienne(w=[0.0, 0.0, 0.5], sigma=1.0, seed=1)   # mu = 0.5 * vitesse
    obs, fini = env.reset(), False
    O, A, LP, R = [], [], [], []
    while not fini:
        a, lp = politique.agir(obs)
        O.append(obs); A.append(a); LP.append(lp)
        obs, r, fini = env.step(a)
        R.append(r)
    O, A, LP, R = map(np.array, (O, A, LP, R))
    print("\n4) Ce qu'on garde d'un épisode :")
    print(f"   observations {O.shape}, actions {A.shape}, log_probs {LP.shape}, récompenses {R.shape}")
    print(f"   somme des récompenses : {R.sum():.1f}")
    print(f"   actions tirées (5 premières) : {A[:5].round(2)}")
    print(f"   leurs log_probs               : {LP[:5].round(2)}")