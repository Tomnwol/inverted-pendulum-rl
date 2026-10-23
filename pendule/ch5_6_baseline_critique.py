"""Chapitres 5 + 6 : réduire le bruit du gradient (baseline, avantage) puis apprendre V(s).

Utilisation :
    python ch5_6_baseline_critique.py              -> parties A et B (mesures)
    python ch5_6_baseline_critique.py train raw 0 500  -> REINFORCE, graine 0, 500 itérations
    python ch5_6_baseline_critique.py train mc 0 500   -> avantage = G - V(s)
    python ch5_6_baseline_critique.py train td 0 500   -> avantage = r + gamma*V(s') - V(s)
"""
import sys
import time
import numpy as np

N, T, GAMMA = 16, 200, 0.95
PALIERS = (1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000)


# ============ PARTIE A : la baseline sur le cas à deux actions ============

# L'idée centrale expliquée:
# Supposons une moyenne de -600, et qu'une passe nous donne un retour de -550
# C'est une amélioration, mais comme c'est négatif, jusqu'au chapitre 4, on punissait le système
# En moyenne, ça marche, mais on sent que c'est très bruité
# L'idée est alors d'av

def partie_A():
    """A rapporte +1 en moyenne, B rapporte 0 (bruit de 1). p = P(A) = 0.5."""
    rng = np.random.default_rng(0)
    n, p = 400_000, 0.5
    is_A = rng.random(n) < p
    score = np.where(is_A, 1 - p, -p)                      # d log pi / d theta
    R = np.where(is_A, 1.0, 0.0) + rng.standard_normal(n)

    print("A) Estimateur du gradient : moyenne et écart-type de (R - b) * score")
    print("   (la vraie valeur du gradient est 0.25)")
    cas = [("récompenses telles quelles, b = 0", R, 0.0),
           ("toutes décalées de -5, b = 0", R - 5, 0.0),
           ("toutes décalées de -5, b = moyenne (-4.5)", R - 5, np.mean(R - 5)),
           ("non décalées, b = moyenne (0.5)", R, np.mean(R))]
    for nom, r, b in cas:
        g = (r - b) * score
        print(f"   {nom:45s} | moyenne {g.mean():+.3f} | écart-type {g.std():.3f}")
    print(f"   E[score] = {score.mean():+.4f}  -> soustraire une constante ne change pas la moyenne")


# ============ outils communs pour le pendule ==============================
def construire(seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)          # AVANT la création des réseaux (initialisation reproductible)
    from torch.distributions import Normal
    from pendule_env import Pendule

    def mlp():
        return nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    actor, critic = mlp(), mlp()

    def prep(o):
        o = torch.tensor(np.array(o), dtype=torch.float32)
        o[..., 2] /= 8
        return o

    def collecter(envs, log_std):
        obs = np.array([e.reset() for e in envs])
        O, A, R = [], [], []
        with torch.no_grad():
            for t in range(T): #200 pas
                o = prep(obs)
                a = Normal(actor(o).squeeze(-1), log_std.exp()).sample()
                O.append(o); A.append(a)
                pas = [e.step(x) for e, x in zip(envs, a.numpy())]
                obs = np.array([p[0] for p in pas])
                R.append(torch.tensor([p[1] for p in pas], dtype=torch.float32))
        O, A, R = map(torch.stack, (O, A, R))
        return O, A, R, prep(obs)

    def retours(R):                                         # G[t] = r[t] + gamma * G[t+1]
        G, suite = torch.zeros_like(R), torch.zeros(R.shape[1])
        for t in reversed(range(T)):
            suite = R[t] + GAMMA * suite
            G[t] = suite
        return G

    return torch, nn, Normal, Pendule, actor, critic, prep, collecter, retours


# ============ PARTIE B : à quel point chaque estimateur est-il bruité ? =====
def partie_B(seed=0):
    torch, nn, Normal, Pendule, actor, critic, prep, collecter, retours = construire(seed)
    log_std = torch.zeros(1)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)
    envs = [Pendule(seed=1000 + i) for i in range(N)]

    # On fige la politique (non entraînée) et on apprend V pour elle, par régression sur G
    for _ in range(150):
        O, A, R, _ = collecter(envs, log_std)
        G, Of = retours(R * 0.1).reshape(-1), O.reshape(-1, 3)
        for _ in range(5):
            perte = 0.5 * ((critic(Of).squeeze(-1) - G) ** 2).mean()
            opt_c.zero_grad(); perte.backward(); opt_c.step()

    def grad_vec(O, A, adv):
        actor.zero_grad()
        lp = Normal(actor(O.reshape(-1, 3)).squeeze(-1), log_std.exp()).log_prob(A.reshape(-1))
        (-(lp * adv.reshape(-1)).mean()).backward()
        return torch.cat([p.grad.flatten() for p in actor.parameters()]).clone()

    def estimations():
        O, A, R, o_last = collecter(envs, log_std)
        R = R * 0.1
        G = retours(R)
        with torch.no_grad():
            V = critic(O).squeeze(-1)
            V_suivant = torch.cat([V[1:], critic(o_last).squeeze(-1)[None]], 0)
            td = R + GAMMA * V_suivant - V

        # ci dessous, le G.mean représente la baseline, qui est la moyenne des retours
        return {"raw (G)": grad_vec(O, A, G),
                "G - moyenne": grad_vec(O, A, G - G.mean()),
                "G - V(s)": grad_vec(O, A, G - V),
                "r + gV(s') - V(s)": grad_vec(O, A, td)}

    ref = torch.stack([estimations()["raw (G)"] for _ in range(40)]).mean(0)   # direction de référence
    tests = [estimations() for _ in range(20)]
    print(f"\nB) graine {seed} : gradient estimé sur 20 lots indépendants (16 épisodes chacun), politique figée.")
    print("   Cosinus avec une direction de référence (moyenne de 40 autres lots 'raw') :")
    for nom in tests[0]:
        g = torch.stack([e[nom] for e in tests])
        cos = torch.nn.functional.cosine_similarity(g, ref[None], dim=1)
        print(f"   {nom:18s} | norme moyenne {g.norm(dim=1).mean():.3f} "
              f"| cosinus : moyen {cos.mean():+.2f}, min {cos.min():+.2f}, max {cos.max():+.2f}")


# ============ PARTIE C : entraînement avec le mode choisi ===================
def partie_C(mode, n_iter=500, lr_actor=3e-3, pas_critique=10, seed=0):
    # mode : "raw" (avantage = G), "mc" (G - V) ou "td" (r + γV(s') - V). Le reste, ce sont les réglages.
    
    # Deux réseaux de même forme. L'acteur sort un couple moyen μ ; le critique sort un retour prédit V(s).
    torch, nn, Normal, Pendule, actor, critic, prep, collecter, retours = construire(seed)
    # Paramètre appris, séparé du réseau : σ = exp(log_std) est la largeur du tirage autour de μ.
    log_std = nn.Parameter(torch.zeros(1))          # au départ log_std = 0, donc σ = 1
    # Un pas met à jour les poids de l'acteur et log_std ensemble.
    opt_a = torch.optim.Adam(list(actor.parameters()) + [log_std], lr=lr_actor)
    # Optimiseur du critique seul. Pas de 1e-3, plus petit que celui de l'acteur (3e-3).
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)
    # 16 pendules en parallèle. Graines distinctes : des positions de départ différentes.
    envs = [Pendule(seed=i + 1) for i in range(N)]

    # Score de la politique déterministe : on envoie μ, sans tirer de couple au hasard.
    def evaluer(n=50):
        # Graine fixe : les 50 départs sont identiques à chaque appel, les scores se comparent.
        env, out = Pendule(seed=123), []
        for _ in range(n):                          # un épisode d'évaluation
            # o : état (cos θ, sin θ, vitesse). total : somme des récompenses. fini : 200 pas écoulés.
            o, total, fini = env.reset(), 0.0, False
            while not fini:                         # jusqu'à la fin de l'épisode
                with torch.no_grad():               # lecture seule : ce passage ne sert pas à apprendre
                    # [None] ajoute la dimension de lot (1, 3). .item() lit le scalaire μ, le couple moyen.
                    u = actor(prep(o)[None]).item()
                # step borne u à [-2, 2], avance le pendule, renvoie l'état suivant et la récompense.
                o, r, fini = env.step(u)
                total += r                          # retour non actualisé, échelle réelle (sans le 0.1)
            out.append(total)                       # un score par épisode
        return np.mean(out)                         # moyenne des 50 épisodes

    t0 = time.time()                                # point de départ pour afficher la durée
    print(f"{mode:3s} | iter {0:4d} | éval {evaluer():8.1f} | std action {log_std.exp().item():.2f}")
    for it in range(1, n_iter + 1):                 # 500 itérations : collecte, puis un pas d'acteur
        # 200 pas sur les 16 pendules, en tirant a ~ Normal(μ, σ). Formes : O (200,16,3), A et R (200,16).
        O, A, R, o_last = collecter(envs, log_std) # o_last : état après le dernier couple, forme (16, 3)
        # R.sum(0) somme les 200 récompenses de chaque pendule, .mean() moyenne sur les 16.
        retour_moyen = R.sum(0).mean().item()      # nombre Python, pour l'affichage seulement
        R = R * 0.1                                 # ramène G et V à une échelle tenable pour les pas d'apprentissage
        with torch.no_grad():                       # V est photographié : l'acteur ne dérivera pas à travers le critique
            V = critic(O).squeeze(-1)               # V(s) pour chaque instant, forme (200, 16) : un retour, pas un couple
            V_dernier = critic(o_last).squeeze(-1) # V(s') du dernier pas, forme (16,)
        # G[t] = r[t] + γ r[t+1] + ... Après le pas 200, la suite vaut 0.
        G = retours(R)
        # Décale V d'un cran dans le temps : V_suivant[t] = V(s_{t+1}). On colle V_dernier à la fin.
        V_suivant = torch.cat([V[1:], V_dernier[None]], 0)
        # Cible amorcée : la récompense vraiment vue, plus ce que le critique attribue déjà à l'état suivant.
        cible_td = R + GAMMA * V_suivant

        if mode == "raw":
            # Chapitre 4 : log π est multiplié par G. cible = None, le critique ne sera pas mis à jour.
            avantage, cible = G, None
        elif mode == "mc":
            # Avantage = G - V(s) : ce couple a-t-il fait mieux que le retour habituel de cet état ?
            # Le critique, lui, apprend à prédire G.
            avantage, cible = G - V, G
        elif mode == "td":
            # Avantage = r + γ V(s') - V(s). Le critique apprend vers r + γ V(s'), pas vers le retour complet G.
            avantage, cible = cible_td - V, cible_td
        else:
            raise ValueError("mode : raw, mc ou td")

        # --- acteur : un pas de REINFORCE. L'avantage pondère, il n'est pas dérivé. ---
        Of, Af = O.reshape(-1, 3), A.reshape(-1)    # 200×16 transitions à plat : Of (3200, 3), Af (3200,)
        # log π(a|s) de chaque couple joué, sous la gaussienne du moment (centre μ, largeur σ).
        logp = Normal(actor(Of).squeeze(-1), log_std.exp()).log_prob(Af)
        # Minimiser −moyenne(log π × avantage) augmente la probabilité des couples à avantage positif.
        perte = -(logp * avantage.reshape(-1)).mean()
        opt_a.zero_grad()                           # efface le gradient de l'itération précédente
        perte.backward()                            # remplit le gradient des poids de l'acteur et de log_std
        opt_a.step()                                # déplace μ et σ dans le sens qui monte l'avantage
        with torch.no_grad():                       # borne imposée, pas une opération à dériver
            log_std.clamp_(-2.0, 0.5)               # σ reste entre exp(-2) ≈ 0.14 et exp(0.5) ≈ 1.65

        # --- critique : rapproche V(s) de la cible. Entièrement sauté en mode raw. ---
        if cible is not None:
            cible = cible.reshape(-1)               # 3200 nombres figés, un par transition
            for _ in range(pas_critique):           # 10 pas de critique pour un seul pas d'acteur
                # On fait 10 passes car chaque petit pas re-mesure la pente après le déplacement.
                # Un seul grand pas parierait sur la pente du départ et dépasserait la cible.
                # critic(Of) produit un nouveau V(s), et perte_c est l'écart entre ce V(s) et la cible.
                perte_c = 0.5 * ((critic(Of).squeeze(-1) - cible) ** 2).mean()
                opt_c.zero_grad()                   # gradient du critique seul
                perte_c.backward()                  # dérivée de l'écart quadratique (V - cible)
                opt_c.step()                        # V(s) se rapproche de G (mc) ou de r + γ V(s') (td)

        if it in PALIERS:
            # Retour exploratoire du lot, retour déterministe, σ courant, secondes écoulées.
            print(f"{mode:3s} | iter {it:4d} | retour (entraînement) {retour_moyen:8.1f} "
                  f"| éval {evaluer():8.1f} | std action {log_std.exp().item():.2f} | {time.time() - t0:3.0f}s")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "train":
        partie_C(sys.argv[2],
                 n_iter=int(sys.argv[4]) if len(sys.argv) > 4 else 500,
                 seed=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else:
        partie_A()
        for graine in (0, 1, 2):
            partie_B(graine)