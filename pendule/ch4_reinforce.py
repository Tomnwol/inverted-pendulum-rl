import sys
import time
import numpy as np

PALIERS = (1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000)


# ======================= PARTIE A : un cas minuscule =======================
# Deux actions : A rapporte en moyenne +1, B rapporte en moyenne 0 (récompenses bruitées).
# Politique : P(A) = p = sigmoide(theta). Un seul paramètre : theta.

def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def jouer(theta, rng, n=1):
    p = sigmoide(theta)
    is_A = rng.random(n) < p                         # tirage de l'action (True = A)
    recompense = np.where(is_A, 1.0, 0.0) + rng.standard_normal(n)
    score = np.where(is_A, 1 - p, -p)                # d log pi(a) / d theta
    return recompense, score


def partie_A():
    rng = np.random.default_rng(0)

    # 1) La formule : grad J = moyenne de [ récompense * d log pi / d theta ]
    theta = 0.0
    p = sigmoide(theta)
    exact = p * (1 - p)                               # J(theta) = p  =>  dJ/dtheta = p(1-p)
    R, S = jouer(theta, rng, n=200_000)
    print(f"1) gradient exact de J en theta=0      : {exact:.4f}")
    print(f"   estimé par la moyenne de R * score  : {np.mean(R * S):.4f}   (200 000 tirages)")

    # 2) L'apprentissage : une action tirée, une mise à jour
    print("\n2) Apprentissage : theta <- theta + lr * R * score   (lr = 0.1)")
    for graine in range(3):
        rng = np.random.default_rng(graine)
        theta, trace = 0.0, []
        for pas in range(1, 2001):
            R, S = jouer(theta, rng)
            theta += 0.1 * R[0] * S[0]
            if pas in (100, 500, 2000):
                trace.append(f"pas {pas}: P(A)={sigmoide(theta):.2f}")
        print(f"   graine {graine} | " + " | ".join(trace))


# ======================= PARTIE B : le pendule ==============================
def partie_B(n_iter=5000, lr=3e-3, seed=0):
    import torch
    import torch.nn as nn
    from torch.distributions import Normal
    from pendule_env import Pendule

    N, T, GAMMA = 16, 200, 0.95
    torch.manual_seed(seed)
    actor = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    log_std = nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam(list(actor.parameters()) + [log_std], lr=lr)

    def prep(o):
        observations = torch.tensor(np.array(o), dtype=torch.float32)
        observations[..., 2] /= 8
        return observations

    def evaluer(n=50):                                # politique déterministe
        # On évalue la politique sur 50 pendules pour voir l'amélioration
        env, retours = Pendule(seed=123), []
        for _ in range(n):
            o, total, fini = env.reset(), 0.0, False
            while not fini:
                with torch.no_grad():
                    u = actor(prep(o)[None]).item()
                o, r, fini = env.step(u)
                total += r
            retours.append(total)
        return np.mean(retours)

    envs = [Pendule(seed=i + 1) for i in range(N)] # On a 16 pendules qui jouent en parallèle
    t0 = time.time()
    print(f"iter {0:4d} | éval {evaluer():8.1f} | std action {log_std.exp().item():.2f}")
    for it in range(1, n_iter + 1): # On fait 5000 itérations
        # Etape 1) collecter des données
        obs = np.array([e.reset() for e in envs]) # on remet les pendules a des positions aleatoires
        O, A, R = [], [], []
        with torch.no_grad(): # On ne calcule pas le gradient ici
            for t in range(T): # On fait 200 pas de temps pour chaque pendule
                o = prep(obs)
                # Etape importante, basé sur l'observatiion actuelle, on choisit pour chaque pendule une valeur autour de la moyenne actor(o)
                # .sample permet de choisir une valeur autour de la distribution normale.
                a = Normal(actor(o).squeeze(-1), log_std.exp()).sample() #le réseau donne un mu pour chacun des 16pendules et on ajoute un bruit gaussien
                O.append(o); A.append(a)
                pas = [e.step(x) for e, x in zip(envs, a.numpy())] # on avance chaque pendule
                obs = np.array([p[0] for p in pas])
                R.append(torch.tensor([p[1] for p in pas], dtype=torch.float32))
        O, A, R = map(torch.stack, (O, A, R))
        retour_moyen = R.sum(0).mean().item() #uniquement pour l'affichage

        # 2) Retours cumulés G[t] (chapitre 1), récompenses mises à l'échelle
        R = R * 0.1
        G, suite = torch.zeros(T, N), torch.zeros(N)
        for t in reversed(range(T)):
            suite = R[t] + GAMMA * suite
            G[t] = suite # R étant <= 0, Les valeurs de G sont négatives également

        # 3) REINFORCE : loss = - moyenne de [ log pi(a|s) * G ]  puis backward
        # On transforme les 200 pas x 16 pendules en 3200 couples (observations, actions, gains)
        O, A, G = O.reshape(-1, 3), A.reshape(-1), G.reshape(-1)
        # logp = log(π(a∣s)) répond à la question : dans l'état s, quelle est la probabilité que je joue l'action a
        # s est l'observation (parfois appelé O)
        logp = Normal(actor(O).squeeze(-1), log_std.exp()).log_prob(A) 
        
        # Plus G est négatif, plus le système a fait une mauvaise action, donc plus la loss est élevée (mauvais)
        # Mais la valeur loss n'a pas directement de sens.
        # Une seule chose est vraiment utile a comprendre, c'est que G permet de pondéré s'il faut, ou non, imiter à nouveau cette action dans le futur
        
        """
        Si on enlevait le G, il resterait -logp.mean(). 
        C'est la loss de l'apprentissage supervisé, qui dirait « imite les actions que tu viens de jouer ». 
        En soi, ça n'apprend rien d'utile.
        Avec G, chaque action est pondérée par sa qualité. On imite fortement les actions qui ont mené à un bon 𝐺
        et on s'éloigne de celles qui ont mené à un mauvais G.
        
        Autre façon de voir le mécanisme :
        Pour chaque pas, la mise à jour pousse les poids de :
        G⋅∇logπ(a|s)

        ∇logπ(a|s) indique dans quelle direction déplacer les poids pour rendre l'action a plus probable.
        G multiplie cette direction :
        si 
        G est positif, on va dans cette direction, et l'action devient plus probable ;
        si 
        G est négatif, on va dans la direction opposée, et l'action devient moins probable ;
        plus 

        |G| est grand, plus le pas est fort.
        Aucun mécanisme ne trie ou n'ignore les actions. Chaque action reçoit un vote, dont le poids est fixé par G.
        Le piège du pendule : ici, 

        G est toujours négatif, donc toutes les actions sont repoussées mais en moyenne ça fonctionne. 
        """
        
        loss = -(logp * G).mean()
        # on initialise les gradients a 0
        opt.zero_grad()
        # loss.backward() permet de calculer le gradient de la loss par rapport aux paramètres du modèle
        loss.backward()
        opt.step()
        # limite le std pour éviter qu'il tombe a 0 ou qu'il devienne trop grand
        with torch.no_grad():
            log_std.clamp_(-2.0, 0.5)

        if it in PALIERS:
            print(f"iter {it:4d} | retour (entraînement) {retour_moyen:8.1f} "
                  f"| éval {evaluer():8.1f} | std action {log_std.exp().item():.2f} | {time.time() - t0:3.0f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        partie_B(n_iter=int(sys.argv[1]))          # python ch4_reinforce.py 500
    else:
        partie_A()
        print("\n--- Partie B : REINFORCE sur le pendule ---")
        partie_B()