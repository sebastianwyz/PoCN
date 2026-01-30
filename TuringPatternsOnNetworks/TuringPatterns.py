import networkx as nx
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
#importiamo odeint
from scipy.integrate import odeint
a = 35
b = 16
c = 9
d = 0.4  # 2/5

u_bar = 5
v_bar = 10


def f(u, v):
    df_du = (a + b*u - u**2)/c - v + ((b -2*u)/c)*u
    df_dv = -u
    dg_du = v
    dg_dv = (u- (1+d*v)) - d*v
    return np.array([[df_du, df_dv],[dg_du, dg_dv]])


def lambda_c(fu, fv, gu, gv, sigma, epsilon, lambdaa):
    return ((1/2)* ((fu + gv) + (1+sigma)*epsilon*lambdaa + [4*fu*gv + (fu - gv + (1-sigma)*epsilon*lambdaa)**2]**0.5 ), (1/2)* ((fu + gv) + (1+sigma)*epsilon*lambdaa - [4*fu*gv + (fu - gv + (1-sigma)*epsilon*lambdaa)**2]**0.5))


def sigma_c(fu, fv, gu, gv, epsilon, lambdaa):
    return ((fu*gv - 2*fv*gu + 2*(fv*gu*(fv*gu - fu*gv))**0.5)/ fu**2)


def generate_scale_free_network(N, m):
    G = nx.barabasi_albert_graph(N, m)
    return G

def laplacian_matrix(A):
    # compute L_ij = A_ij - k_i*dirac_delta_ij
    A = nx.to_numpy_array(A)
    L = np.diag(np.sum(A, axis=1))  - A
    return L

def calculate_eigenvalues(L):
    eigenvalues, _ = eigh(L)
    return eigenvalues

def caluclate_eigenvectors(L):
    _, eigenvectors = eigh(L)
    return eigenvectors



import networkx as nx
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
a = 35
b = 16
c = 9
d = 0.4  # 2/5
u_bar = 5
v_bar = 10

def f(u, v):
    df_du = (a + b*u - u**2)/c - v + ((b -2*u)/c)*u
    df_dv = -u
    dg_du = v
    dg_dv = (u- (1+d*v)) - d*v
    return np.array([[df_du, df_dv],[dg_du, dg_dv]])


def lambda_c(fu, fv, gu, gv, sigma, epsilon, lambdaa):
    return ((1/2)* ((fu + gv) + (1+sigma)*epsilon*lambdaa + [4*fu*gv + (fu - gv + (1-sigma)*epsilon*lambdaa)**2]**0.5 ), (1/2)* ((fu + gv) + (1+sigma)*epsilon*lambdaa - [4*fu*gv + (fu - gv + (1-sigma)*epsilon*lambdaa)**2]**0.5))


def sigma_c(fu, fv, gu, gv, epsilon, lambdaa):
    return ((fu*gv - 2*fv*gu + 2*(fv*gu*(fv*gu - fu*gv))**0.5)/ fu**2)


def generate_scale_free_network(N, m):
    G = nx.barabasi_albert_graph(N, m)
    return G

def laplacian_matrix(A):
    A = nx.to_numpy_array(A)
    L = np.diag(np.sum(A,axis=1))  -A
    return L

def calculate_eigenvalues(L):
    eigenvalues, _ = eigh(L)
    return eigenvalues

def calculate_eigenvectors(L):
    _, eigenvectors = eigh(L)
    return eigenvectors

J = f(u_bar, v_bar)
fu = J[0,0]
fv = J[0,1]
gu = J[1,0]
gv = J[1,1]

def growth_rate(eps_Lambda, sigma):
    term1 = fu+gv + (1+sigma)*eps_Lambda
    discriminant = (fu -gv +(1 -sigma) *eps_Lambda)**2 + 4 *fv*gu
    sqrt_disc = np.sqrt(discriminant + 0j)
    lambda_plus = 0.5*(term1 +sqrt_disc)
    return np.real(lambda_plus)  

def calculate_sigma_c():
    return (fu *gv -2*fv *gu + 2 *np.sqrt(fv *gu *(fv *gu -fu *gv))) /fu**2

N = 200
m = 10
#G = generate_scale_free_network(N, m)

# er matrix
import networkx as nx
#G = nx.erdos_renyi_graph(N, p=0.1)
#G= nx.stochastic_block_model([N//2, N//2], [[0.05, 0.01],[0.01, 0.05]])
G= generate_scale_free_network(N, m)
L_standard = laplacian_matrix(G)
L_article = -L_standard

Lambda, eigvecs = eigh(L_article)
sort_idx = np.argsort(Lambda)[::-1]  
Lambda = Lambda[sort_idx]
eigvecs = eigvecs[:, sort_idx]

degrees = np.sum(nx.to_numpy_array(G), axis=1)
node_order = np.argsort(-degrees)

sigma_c_val = calculate_sigma_c()

def calculate_Lambda_c(eps, sigma):
    numerator = (fu - gv) * sigma - np.sqrt(np.abs(fv) * gu * sigma) * (sigma + 1)
    denominator = eps * sigma * (sigma - 1)
    return numerator / denominator


def plot_figure1():
    eps_values = [0.425, 0.165, 0.060]
    sigma_values = [15.0, sigma_c_val, 16.0]
    min_Lambda = min(Lambda)
    Lambda_range = np.linspace(-0.001, min_Lambda, 1000)  
    ln_neg_Lambda = np.log(-Lambda_range)  
    
    plt.figure()
    labels = ['1', '2', '3', '4']
    label_idx = 0
    for eps in eps_values:
        for sigma in sigma_values:
            if sigma == sigma_c_val or (eps == 0.060 and sigma != sigma_c_val):
                lambdas = growth_rate(eps * Lambda_range, sigma)
                ls = '-' if sigma == sigma_c_val else '--'
                plt.plot(ln_neg_Lambda, lambdas, linestyle=ls, label=f'ε={eps}, σ={sigma:.1f}')
                
                if sigma == sigma_c_val:
                    Lambda_c = calculate_Lambda_c(eps, sigma)
                    ln_neg_Lambda_c = np.log(-Lambda_c)
                    lambda_at_c = 0  # Al threshold, λ=0
                    plt.plot(ln_neg_Lambda_c, lambda_at_c, 'ro')
                    plt.text(ln_neg_Lambda_c, lambda_at_c + 0.01, labels[label_idx], fontsize=24)
                    label_idx += 1
    
    plt.xlabel('ln(-Λ)', fontsize=24)
    plt.ylabel('λ', fontsize=24)
    plt.legend()
    plt.xlim(0, 4)  
    plt.ylim(-0.5, 0.2)
    #plt.title('Linear growth rates vs ln(-Λ)', fontsize=24)
    plt.show()

def find_critical_mode(eps, sigma):
    lambdas = growth_rate(eps * Lambda, sigma)
    alpha_c = np.argmax(lambdas[1:]) + 1  
    return alpha_c

def plot_figure2_ab():
    eps_values = [0.060, 0.425]  # a: small eps, b: large eps
    labels = ['a', 'b']
    ln_i = np.log(np.arange(1, N + 1))
    for idx, eps in enumerate(eps_values):
        alpha_c = find_critical_mode(eps, sigma_c_val)
        phi = eigvecs[:, alpha_c]
        phi_sorted = phi[node_order]
        k_sorted = degrees[node_order]
        plt.figure()
        ax1 = plt.gca()
        ax1.scatter(ln_i, phi_sorted, color='b', marker='o', label='φ_i')  # Cambiato in scatter per "punti per nodo"
        ax1.set_xlabel(r'$\ln i$', fontsize=24)
        #ax1.set_ylabel('φ_i', fontsize=24)
        ax1.set_ylabel(r'$\varphi_i$', fontsize=24)
        ax2 = ax1.twinx()
        ax2.step(ln_i, k_sorted, 'g-', where='post', label='k_i')
        ax2.set_ylabel(r'$k_i$', fontsize=24)
        plt.title(f'Critical mode for ε={eps}', fontsize=24)
        plt.show()

# Figure 2c/d: Visualizzazione rete con nodi colorati
def plot_figure2_cd():
    eps_values = [0.060, 0.425]  # c: small eps, d: large eps
    labels = ['c', 'd']
    threshold = 0.1  # Come nell'articolo: ≥0.1 red, ≤-0.1 blue, else yellow
    pos = nx.spring_layout(G)  # Layout per disegno rete (centro hub)
    for idx, eps in enumerate(eps_values):
        alpha_c = find_critical_mode(eps, sigma_c_val)
        phi = eigvecs[:, alpha_c]
        colors = []
        for val in phi:
            if val >= threshold:
                colors.append('red')
            elif val <= -threshold:
                colors.append('blue')
            else:
                colors.append('yellow')
        plt.figure()
        nx.draw(G, pos, node_color=colors, with_labels=False, node_size=50)
        plt.title(f'Network view for ε={eps}')
        plt.show()


def plot_figure3():
    threshold = 0.1  # Come nell'articolo per "differentiated" nodes
    # Per a: N=200, <k>=10 (già G, degrees, Lambda, eigvecs)
    # Per b: Genera nuova rete N=1000, <k>=20
    N_large = 1000
    m_large = 20
    G_large = generate_scale_free_network(N_large, m_large)
    #G_large = nx.erdos_renyi_graph(N_large,p=0.1) #generate_scale_free_network(N_large, m_large)
    #G_large= nx.stochastic_block_model([N_large//2, N_large//2], [[0.05, 0.01],[0.01, 0.05]])

    L_standard_large = laplacian_matrix(G_large)
    L_article_large = -L_standard_large
    Lambda_large, eigvecs_large = eigh(L_article_large)
    sort_idx_large = np.argsort(Lambda_large)[::-1]
    Lambda_large = Lambda_large[sort_idx_large]
    eigvecs_large = eigvecs_large[:, sort_idx_large]
    degrees_large = np.sum(nx.to_numpy_array(G_large), axis=1)

    # Funzione helper per calcolare density per una rete
    def compute_density(degrees_in, Lambda_in, eigvecs_in):
        unique_k = np.unique(degrees_in)
        unique_k_sorted = np.sort(unique_k[unique_k > 0])  # Skip k=0 se presente
        ln_k = np.log(unique_k_sorted)
        k_groups = {k: np.where(degrees_in == k)[0] for k in unique_k_sorted}
        
        density = np.zeros((len(Lambda_in) - 1, len(unique_k_sorted)))
        for alpha in range(1, len(Lambda_in)):  # Skip modo uniforme α=0
            phi = eigvecs_in[:, alpha]
            for k_idx, k in enumerate(unique_k_sorted):
                nodes_k = k_groups[k]
                if len(nodes_k) > 0:
                    num_diff = np.sum(np.abs(phi[nodes_k]) >= threshold)
                    density[alpha - 1, k_idx] = num_diff / len(nodes_k)
        
        ln_neg_Lambda = np.log(-Lambda_in[1:])
        X, Y = np.meshgrid(ln_k, ln_neg_Lambda)
        return X, Y, density
    
    # Computa per a e b
    X_a, Y_a, density_a = compute_density(degrees, Lambda, eigvecs)
    X_b, Y_b, density_b = compute_density(degrees_large, Lambda_large, eigvecs_large)
    
    # Plot a e b side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    im_a = axs[0].pcolormesh(X_a, Y_a, density_a, cmap='hot', vmin=0, vmax=0.8)
    axs[0].set_xlabel('ln k', fontsize=24)
    axs[0].set_ylabel('ln (-Λ)', fontsize=24)
    axs[0].set_title('N=200, <k>=10')
    
    im_b = axs[1].pcolormesh(X_b, Y_b, density_b, cmap='hot', vmin=0, vmax=0.8)
    axs[1].set_xlabel('ln k', fontsize=24)
    #axs[1].set_ylabel('ln (-Λ)', fontsize=24)
    axs[1].set_title('N=1000, <k>=20')
    
    fig.colorbar(im_b, ax=axs, label='Density')
    #plt.suptitle(' Localization of Laplacian eigenvectors', fontsize=24)
    plt.show()


# Funzioni locali f e g per Mimura-Murray
def f_loc(u, v):
    return u * ((a + b * u - u**2) / c - v)

def g_loc(u, v):
    return v * (u - 1 - d * v)

# RHS per ODE
def rhs(state, t, L, eps, sigma):
    u = state[0::2]
    v = state[1::2]
    du = f_loc(u, v) + eps * (L @ u)
    dv = g_loc(u, v) + sigma * eps * (L @ v)
    dstate = np.zeros_like(state)
    dstate[0::2] = du
    dstate[1::2] = dv
    return dstate

# Figure 4: Nonlinear evolution
def plot_figure4():
    np.random.seed(42)  # Per riproducibilità
    eps = 0.16  #0.12
    sigma = 15.6
    pert_scale = 0.01
    t_max = 3000
    t_early = 200
    t_late = 3000
    #N = 1000
    state_uniform = np.tile([u_bar, v_bar], N)
    state0 = state_uniform + pert_scale * np.random.randn(2 * N)
    t = np.linspace(0, t_max, 903) 
    state_t = odeint(rhs, state0, t, args=(L_article, eps, sigma))
    
    alpha_c = find_critical_mode(eps, sigma)
    phi_sorted = eigvecs[:, alpha_c][node_order]
    
    idx_early = np.argmin(np.abs(t - t_early))
    idx_late = np.argmin(np.abs(t - t_late))
    
    u_early = state_t[idx_early, 0::2][node_order]
    u_late = state_t[idx_late, 0::2][node_order]
    k_sorted = degrees[node_order]
    i = np.arange(1, N + 1)
    
    # a: Critical mode
    plt.figure()
    plt.scatter(i, phi_sorted, color='b', marker='o')
    plt.plot(i, phi_sorted, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('i', fontsize=24)
    plt.ylabel(r'$\varphi_i$', fontsize=24)
    #plt.title('Critical mode', fontsize=24)
    plt.show()
    
    # b: Early stage
    plt.figure()
    plt.scatter(i, u_early, color='b', marker='o')
    plt.plot(i, u_early, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('i', fontsize=24)
    plt.ylabel(r'$u_i$', fontsize=24)
    plt.title('Early evolution (t=200)', fontsize=24)
    plt.show()
    
    # c: Stationary
    plt.figure()
    plt.scatter(i, u_late, color='b', marker='o')
    plt.plot(i, u_late, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('i', fontsize=24)
    plt.ylabel(r'$u_i$', fontsize=24)
    plt.title('Stationary pattern (t=1500)', fontsize=24)
    plt.show()
    
    # d: Degrees
    plt.figure()
    plt.step(i, k_sorted, 'g-', where='post')
    plt.xlabel('i', fontsize=24)
    plt.ylabel(r'$k_i$', fontsize=24)
    #plt.title('Node degrees', fontsize=24)
    plt.show()


# Esegui i plot
plot_figure1()
plot_figure2_ab()
plot_figure3()
plot_figure4()



from scipy.integrate import odeint

def plot_perturbation_analysis():
    """
    Analizza l'effetto della rimozione di nodi sulla dinamica del pattern di Turing.
    Confronta rimozione casuale vs rimozione di hub.
    """
    np.random.seed(42)
    eps = 0.16
    sigma = 15.6
    pert_scale = 0.01
    t_differentiation = 500  # Tempo per permettere differenziazione
    t_after_removal = 1000   # Tempo dopo rimozione nodi
    p = int(N/5)  # Numero di nodi da rimuovere
    
    state_uniform = np.tile([u_bar, v_bar], N)
    state0 = state_uniform + pert_scale * np.random.randn(2 * N)
    
    t_phase1 = np.linspace(0, t_differentiation, 200)
    state_phase1 = odeint(rhs, state0, t_phase1, args=(L_article, eps, sigma))
    state_differentiated = state_phase1[-1, :]
    
    u_before = state_differentiated[0::2]
    
    # Caso 1: Rimozione casuale
    np.random.seed(123)
    nodes_random = np.random.choice(N, p, replace=False)
    G_random = G.copy()
    G_random.remove_nodes_from(nodes_random)
    
    remaining_nodes_random = sorted(set(range(N)) - set(nodes_random))
    mapping_random = {old: new for new, old in enumerate(remaining_nodes_random)}
    G_random_reindexed = nx.relabel_nodes(G_random, mapping_random)
    
    L_random = laplacian_matrix(G_random_reindexed)
    L_article_random = -L_random
    
    # Stato iniziale per rete perturbata
    state_random = np.zeros(2 * len(remaining_nodes_random))
    for new_idx, old_idx in enumerate(remaining_nodes_random):
        state_random[2*new_idx] = state_differentiated[2*old_idx]
        state_random[2*new_idx + 1] = state_differentiated[2*old_idx + 1]
    
    # Evoluzione dopo rimozione random
    t_phase2 = np.linspace(0, t_after_removal, 151)
    state_random_evolved = odeint(rhs, state_random, t_phase2, 
                                   args=(L_article_random, eps, sigma))
    
    # Caso 2: Rimozione hub (nodi con grado più alto)
    nodes_hub = node_order[:p]  # Primi p nodi ordinati per grado
    G_hub = G.copy()
    G_hub.remove_nodes_from(nodes_hub)
    
    remaining_nodes_hub = sorted(set(range(N)) - set(nodes_hub))
    mapping_hub = {old: new for new, old in enumerate(remaining_nodes_hub)}
    G_hub_reindexed = nx.relabel_nodes(G_hub, mapping_hub)
    
    L_hub = laplacian_matrix(G_hub_reindexed)
    L_article_hub = -L_hub
    
    state_hub = np.zeros(2 * len(remaining_nodes_hub))
    for new_idx, old_idx in enumerate(remaining_nodes_hub):
        state_hub[2*new_idx] = state_differentiated[2*old_idx]
        state_hub[2*new_idx + 1] = state_differentiated[2*old_idx + 1]
    
    state_hub_evolved = odeint(rhs, state_hub, t_phase2, 
                               args=(L_article_hub, eps, sigma))
    
    # Plot risultati
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # a) Pattern prima della rimozione
    i_before = np.arange(1, N + 1)
    axes[0, 0].scatter(i_before, u_before[node_order], c='b', marker='o', s=20)
    axes[0, 0].plot(i_before, u_before[node_order], 'k--', linewidth=0.5)
    axes[0, 0].set_xlabel('Node i ', fontsize=20)
    axes[0, 0].set_ylabel(r'$u_i$', fontsize=20)
    axes[0, 0].set_title(f'Before removal (t={t_differentiation})', fontsize=20)
    axes[0, 0].axvline(p, color='r', linestyle=':', label=f'{p} highest degree nodes')
    axes[0, 0].legend()
    
    # b) Dopo rimozione random
    u_random_final = state_random_evolved[-1, 0::2]
    degrees_random = np.sum(nx.to_numpy_array(G_random_reindexed), axis=1)
    order_random = np.argsort(-degrees_random)
    i_random = np.arange(1, len(remaining_nodes_random) + 1)
    axes[0, 1].scatter(i_random, u_random_final[order_random], c='g', marker='o', s=20)
    axes[0, 1].plot(i_random, u_random_final[order_random], 'k--', linewidth=0.5)
    axes[0, 1].set_xlabel('Node i ', fontsize=20)
    axes[0, 1].set_ylabel(r'$u_i$', fontsize=20)
    axes[0, 1].set_title(f'Random removal (t={t_differentiation + t_after_removal})', fontsize=20)
    
    # c) Dopo rimozione hub
    u_hub_final = state_hub_evolved[-1, 0::2]
    degrees_hub = np.sum(nx.to_numpy_array(G_hub_reindexed), axis=1)
    order_hub = np.argsort(-degrees_hub)
    i_hub = np.arange(1, len(remaining_nodes_hub) + 1)
    axes[1, 0].scatter(i_hub, u_hub_final[order_hub], c='r', marker='o', s=20)
    axes[1, 0].plot(i_hub, u_hub_final[order_hub], 'k--', linewidth=0.5)
    axes[1, 0].set_xlabel('Node i ', fontsize=20)
    axes[1, 0].set_ylabel(r'$u_i$', fontsize=20)
    axes[1, 0].set_title(f'Hub removal (t={t_differentiation + t_after_removal})', fontsize=20)
    
    # d) Confronto evoluzione temporale (media u)
    u_mean_random = np.mean(state_random_evolved[:, 0::2], axis=1)
    u_mean_hub = np.mean(state_hub_evolved[:, 0::2], axis=1)
    axes[1, 1].plot(t_phase2[:40], u_mean_random[:40], 'g-', label='Random removal', linewidth=2)
    axes[1, 1].plot(t_phase2[:40], u_mean_hub[:40], 'r-', label='Hub removal', linewidth=2)
    axes[1, 1].axhline(u_bar, color='k', linestyle=':', label='Uniform state')
    axes[1, 1].set_xlabel('Time after removal (s)', fontsize=20)
    axes[1, 1].set_ylabel('Mean u', fontsize=20)
    axes[1, 1].set_title('Temporal evolution comparison', fontsize=20)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    #plt.suptitle(f'Perturbation Analysis: Removal of {p} nodes', y=1.02, fontsize=20)
    plt.show()
    
    # Stampa statistiche
    print(f"Original network: N={N}, <k>={np.mean(degrees):.2f}")
    print(f"\nRandom removal:")
    print(f"  Remaining nodes: {len(remaining_nodes_random)}")
    print(f"  Mean degree: {np.mean(degrees_random):.2f}")
    print(f"  Pattern amplitude: {np.std(u_random_final):.4f}")
    print(f"\nHub removal:")
    print(f"  Remaining nodes: {len(remaining_nodes_hub)}")
    print(f"  Mean degree: {np.mean(degrees_hub):.2f}")
    print(f"  Pattern amplitude: {np.std(u_hub_final):.4f}")

# Esegui l'analisi
plot_perturbation_analysis()


# Figure 6: Pattern vs mean-field (per σ=15.6 e 30)
from scipy.optimize import fsolve
def plot_figure6():
    eps = 0.12
    sigma_values = [15.6, 30.0]
    t_max = 1000
    pert_scale = 0.001
    
    for sigma in sigma_values:
        # Calcola il pattern
        state_uniform = np.tile([u_bar, v_bar], N)
        state0 = state_uniform + pert_scale * np.random.randn(2 * N)
        t = np.linspace(0, t_max, 101)
        state_t = odeint(rhs, state0, t, args=(L_article, eps, sigma))
        state_ss = state_t[-1]
        u_ss = state_ss[0::2]
        v_ss = state_ss[1::2]
        
        # Ordina secondo i gradi
        u_sorted = u_ss[node_order]
        
        # Plot solo i punti del pattern
        i = np.arange(1, N + 1)
        plt.figure()
        plt.plot(i, u_sorted, 'kx', label='Computed u_i')
        plt.xlabel('i')
        plt.ylabel('u_i')
        plt.legend()
        plt.title(f'Pattern for σ={sigma}')
        plt.show()
N = 250  # Ridotto da 1000 a 200 nodi
#G = generate_scale_free_network(N, m)  # Rigenera la rete
G=nx.erdos_renyi_graph(N, p=0.1)
L_standard = laplacian_matrix(G)  # Ricalcola laplaciana
L_article = -L_standard
# Ricalcola autovalori e autovettori
Lambda, eigvecs = eigh(L_article)
sort_idx = np.argsort(Lambda)[::-1]
Lambda = Lambda[sort_idx]
eigvecs = eigvecs[:, sort_idx]
# Aggiorna gradi
degrees = np.sum(nx.to_numpy_array(G), axis=1)
node_order = np.argsort(-degrees)



plot_figure6()


def rhs(state, t, L, eps, sigma):
    u = state[0::2]
    v = state[1::2]
    du = f_loc(u, v) + eps * (L @ u)
    dv = g_loc(u, v) + sigma * eps * (L @ v)
    dstate = np.zeros_like(state)
    dstate[0::2] = du
    dstate[1::2] = dv
    return dstate


def plot_figure5a_updown():
    """
    Versione completa con sweep up e down (isteresi)
    """
    eps = 0.12
    t_max = 500
    t_points = 51
    pert_scale = 0.01
    
    # Range sigma
    sigma_min = 12.0
    sigma_max = 18.0
    sigma_step = 0.2
    
    # Sweep crescente
    sigma_up = np.arange(sigma_min, sigma_max + sigma_step, sigma_step)
    # Sweep decrescente
    sigma_down = np.arange(sigma_max, sigma_min - sigma_step, -sigma_step)
    
    A_up = []
    A_down = []
    
    # === SWEEP CRESCENTE ===
    print("=== Sweep crescente ===")
    state_prev = np.tile([u_bar, v_bar], N)
    
    for i, sigma in enumerate(sigma_up):
        state0 = state_prev + pert_scale * np.random.randn(2 * N)
        t = np.linspace(0, t_max, t_points)
        
        state_t = odeint(rhs, state0, t, args=(L_article, eps, sigma), 
                        rtol=1e-6, atol=1e-6)
        
        state_ss = state_t[-1]
        u_ss = state_ss[0::2]
        v_ss = state_ss[1::2]
        
        A = np.sqrt(np.sum((u_ss - u_bar)**2 + (v_ss - v_bar)**2))
        A_up.append(A)
        state_prev = state_ss
        
        print(f"[{i+1}/{len(sigma_up)}] σ={sigma:.2f}, A={A:.2f}")
    
    # === SWEEP DECRESCENTE ===
    print("\n=== Sweep decrescente ===")
    # Parti dall'ultimo stato del sweep crescente
    
    for i, sigma in enumerate(sigma_down):
        state0 = state_prev + pert_scale * np.random.randn(2 * N)
        t = np.linspace(0, t_max, t_points)
        
        state_t = odeint(rhs, state0, t, args=(L_article, eps, sigma), 
                        rtol=1e-6, atol=1e-6)
        
        state_ss = state_t[-1]
        u_ss = state_ss[0::2]
        v_ss = state_ss[1::2]
        
        A = np.sqrt(np.sum((u_ss - u_bar)**2 + (v_ss - v_bar)**2))
        A_down.append(A)
        state_prev = state_ss
        
        print(f"[{i+1}/{len(sigma_down)}] σ={sigma:.2f}, A={A:.2f}")
    
    # === PLOT ===

    plt.figure(figsize=(10, 6))
    plt.scatter(sigma_up, A_up, color='b', marker='o', s=50, 
                label='Increasing σ', alpha=0.7)
    plt.scatter(sigma_down, A_down, color='r', marker='s', s=50, 
                label='Decreasing σ', alpha=0.7)
    plt.axvline(x=15.5, color='gray', linestyle='--', linewidth=1.5, 
                label='σ_c ≈ 15.5', alpha=0.5)
    
    plt.xlabel('σ (diffusion ratio)', fontsize=20)
    plt.ylabel('Amplitude A', fontsize=20)
    plt.title('Hysteresis and Multistability', fontsize=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    
    return sigma_up, A_up, sigma_down, A_down

# Parametri modello (globali)
a, b, c, d = 35, 16, 9, 2/5
u_bar, v_bar = 5, 10

# Setup rete (usa i tuoi parametri esistenti o ricrea)
N = 500  # o 1000 per risultati più fedeli all'articolo
m = 10   # per <k> = 20
G = generate_scale_free_network(N, m)
L_standard = laplacian_matrix(G)
L_article = -L_standard

# Calcola autovalori/autovettori se servono
Lambda, eigvecs = eigh(L_article)
sort_idx = np.argsort(Lambda)[::-1]
Lambda = Lambda[sort_idx]
eigvecs = eigvecs[:, sort_idx]

# Esegui
sigma_up, A_up, sigma_down, A_down = plot_figure5a_updown()


from scipy.integrate import odeint

# Figure 6: Pattern vs mean-field bifurcation diagram
def plot_figure6_complete():
    eps = 0.12
    sigma_values = [15.6, 30.0]
    t_max = 1500
    pert_scale = 0.001
    
    # Usa la rete corrente e ricalcola node_order
    current_degrees = np.sum(nx.to_numpy_array(G), axis=1)
    current_node_order = np.argsort(-current_degrees)
    current_N = len(current_degrees)
    
    for sigma in sigma_values:
        # Simula il pattern di Turing
        state_uniform = np.tile([u_bar, v_bar], current_N)
        state0 = state_uniform + pert_scale * np.random.randn(2 * current_N)
        t = np.linspace(0, t_max, 201)
        state_t = odeint(rhs, state0, t, args=(L_article, eps, sigma))
        state_ss = state_t[-1]
        u_ss = state_ss[0::2]
        v_ss = state_ss[1::2]
        
        # Ordina secondo i gradi
        u_sorted = u_ss[current_node_order]
        v_sorted = v_ss[current_node_order]
        
        # Calcola i campi medi globali H^(u) e H^(v)
        k_total = np.sum(current_degrees)
        weights = current_degrees / k_total
        H_u = np.sum(weights * u_ss)
        H_v = np.sum(weights * v_ss)
        
        # Calcola il diagramma di biforcazione mean-field
        # Per ogni valore di beta = eps * k_i, risolvi il sistema mean-field
        k_range = np.linspace(1, np.max(current_degrees), 300)
        beta_range = eps * k_range
        
        u_stable = []
        u_unstable = []
        beta_stable = []
        beta_unstable = []
        
        for beta in beta_range:
            # Sistema mean-field: equazioni (3) dell'articolo
            def mean_field_eq(uv):
                u, v = uv
                eq1 = f_loc(u, v) + beta * (H_u - u)
                eq2 = g_loc(u, v) + sigma * beta * (H_v - v)
                return [eq1, eq2]
            
            # Trova soluzioni multiple provando diversi punti iniziali
            init_guesses = [
                [u_bar, v_bar],
                [u_bar * 0.5, v_bar * 0.5],
                [u_bar * 1.5, v_bar * 1.5],
                [u_bar * 2, v_bar * 2],
                [1, 5],
                [8, 15]
            ]
            
            solutions = []
            for guess in init_guesses:
                try:
                    sol = fsolve(mean_field_eq, guess, full_output=True)
                    if sol[2] == 1:  # Convergenza
                        u_sol, v_sol = sol[0]
                        # Verifica che sia una soluzione unica
                        is_new = True
                        for existing_sol in solutions:
                            if np.abs(u_sol - existing_sol[0]) < 0.01:
                                is_new = False
                                break
                        if is_new and u_sol > 0 and v_sol > 0:
                            solutions.append([u_sol, v_sol])
                except:
                    pass
            
            # Analisi di stabilità per ogni soluzione
            for sol in solutions:
                u_sol, v_sol = sol
                # Jacobiano del sistema mean-field
                J_mf = np.array([
                    [fu + (b - 2*u_sol)/c * u_sol - beta, fv * u_sol],
                    [gu * v_sol, gv - d * v_sol - sigma * beta]
                ])
                eigenvals = np.linalg.eigvals(J_mf)
                
                if np.all(np.real(eigenvals) < 0):
                    # Stabile
                    u_stable.append(u_sol)
                    beta_stable.append(beta)
                else:
                    # Instabile
                    u_unstable.append(u_sol)
                    beta_unstable.append(beta)
        
        # Converti beta in indice i usando la relazione beta = eps * k_i
        k_sorted = current_degrees[current_node_order]
        i = np.arange(1, current_N + 1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        # Pattern computato (croci nere)
        plt.plot(i, u_sorted, 'kx', markersize=6, alpha=0.6, label='Computed pattern')
        
        # Diagramma di biforcazione mean-field
        if beta_stable:
            # Mappa beta su i usando i gradi ordinati
            i_stable = np.interp(beta_stable, eps * k_sorted, i)
            plt.plot(i_stable, u_stable, 'b.', markersize=4, alpha=0.8, label='Stable branch (mean-field)')
        
        if beta_unstable:
            i_unstable = np.interp(beta_unstable, eps * k_sorted, i)
            plt.plot(i_unstable, u_unstable, 'c.', markersize=4, alpha=0.3, label='Unstable branch (mean-field)')
        
        plt.xlabel('i', fontsize=20)
        plt.ylabel('u_i', fontsize=20)
        plt.legend(fontsize=15)
        plt.title(f'Turing pattern vs mean-field (σ={sigma}, ε={eps})', fontsize=20)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

# Esegui
plot_figure6_complete()
