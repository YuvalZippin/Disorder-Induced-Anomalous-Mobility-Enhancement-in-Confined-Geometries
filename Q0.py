import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def test_convergence_detailed():
    """בדיקה מפורטת של התכנסות"""
    
    print("="*60)
    print("בדיקת התכנסות מפורטת - למה w=100 נותן תוצאות שגויות?")
    print("="*60)
    
    # פונקציה לחישוב 3D
    def calculate_3D(w):
        B = 1/6
        sum_term = 0
        for m in range(w):
            cos_term = np.cos(2 * np.pi * m / w)
            denominator = (1 - 4*B*cos_term)**2 - 4*B**2
            if denominator > 0:
                sum_term += denominator**(-0.5)
        
        W_tilde = w**2 / sum_term
        Q0 = 1 - 1/W_tilde
        return Q0, sum_term, W_tilde
    
    # פונקציה לחישוב 2D
    def calculate_2D(w, B_val):
        sum_term = 0
        for m in range(w):
            cos_term = np.cos(2 * np.pi * m / w)
            denominator = (1 - 2*B_val*cos_term)**2 - 4*B_val**2
            if denominator > 0:
                sum_term += denominator**(-0.5)
        
        Q0 = 1 - w/sum_term
        return Q0, sum_term
    
    # בדיקה של ערכי w שונים
    w_values = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    
    print("\n3D Case Analysis:")
    print("w\t\tSum\t\tW̃₁(0)\t\tQ₀*")
    print("-" * 60)
    
    for w in w_values:
        Q0, sum_val, W_tilde = calculate_3D(w)
        print(f"{w:d}\t\t{sum_val:.2e}\t{W_tilde:.2e}\t{Q0:.6f}")
    
    print("\n2D Case Analysis (B=1/4):")
    print("w\t\tSum\t\tQ₀*")
    print("-" * 40)
    
    for w in w_values:
        Q0, sum_val = calculate_2D(w, 1/4)
        print(f"{w:d}\t\t{sum_val:.2f}\t\t{Q0:.6f}")
    
    # חישוב הגבול הרציף
    print("\n" + "="*60)
    print("השוואה עם הגבול הרציף:")
    
    # 3D continuous
    def integrand_3d(kx, ky):
        B0 = 1/6
        denom = (1 - 2*B0*np.cos(kx) - 2*B0*np.cos(ky))**2 - (2*B0)**2
        return 1/np.sqrt(denom) if denom > 0 else 0
    
    val_3d, _ = integrate.nquad(integrand_3d, [[0, 2*np.pi], [0, 2*np.pi]])
    avg_3d = val_3d / (2*np.pi)**2
    Q0_3d_cont = 1 - 1/avg_3d
    
    print(f"3D Continuous: {Q0_3d_cont:.6f}")
    
    # 2D continuous
    def integrand_2d(k):
        B0 = 1/4
        denom = (1 - 2*B0*np.cos(k))**2 - 4*B0**2
        return 1/np.sqrt(denom) if denom > 0 else 0
    
    val_2d, _ = integrate.quad(integrand_2d, 0, 2*np.pi)
    avg_2d = val_2d / (2*np.pi)
    Q0_2d_cont = 1 - 1/avg_2d
    
    print(f"2D Continuous: {Q0_2d_cont:.6f}")
    
    # ניתוח הסיבות לאי-התכנסות
    print("\n" + "="*60)
    print("ניתוח הסיבות לאי-התכנסות:")
    print("="*60)
    
    print("\n1. בעיית קירוב ריימן:")
    print("   הסכום הדיסקרטי הוא קירוב ריימן לאינטגרל")
    print("   עבור w קטן, הקירוב לא מדויק")
    
    print("\n2. בעיית סינגולריות:")
    w_test = 100
    B = 1/6
    problem_points = []
    
    for m in range(w_test):
        cos_term = np.cos(2 * np.pi * m / w_test)
        denominator = (1 - 4*B*cos_term)**2 - 4*B**2
        if denominator < 0.01:  # נקודות קריטיות
            problem_points.append((m, cos_term, denominator))
    
    print(f"   נמצאו {len(problem_points)} נקודות קריטיות עבור w={w_test}")
    if problem_points:
        print("   דוגמאות לנקודות בעייתיות:")
        for i, (m, cos_val, denom) in enumerate(problem_points[:3]):
            print(f"   m={m}: cos={cos_val:.4f}, denom={denom:.6f}")
    
    print("\n3. בעיית נורמליזציה:")
    print("   במקרה 3D: W̃₁(0) = w²/sum")
    print("   כאשר sum גדול מאוד, W̃₁(0) הופך קטן מאוד")
    print("   ואז Q₀* = 1 - 1/W̃₁(0) הופך שלילי וגדול")
    
    return w_values, Q0_3d_cont, Q0_2d_cont

def plot_convergence_analysis():
    """גרף של התכנסות"""
    w_values = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000])
    
    # חישוב ערכים
    Q0_3d_values = []
    Q0_2d_values = []
    
    for w in w_values:
        # 3D
        B = 1/6
        sum_3d = 0
        for m in range(w):
            cos_term = np.cos(2 * np.pi * m / w)
            denominator = (1 - 4*B*cos_term)**2 - 4*B**2
            if denominator > 1e-15:
                sum_3d += denominator**(-0.5)
        
        if sum_3d > 0:
            W_tilde = w**2 / sum_3d
            Q0_3d = 1 - 1/W_tilde
        else:
            Q0_3d = np.nan
        Q0_3d_values.append(Q0_3d)
        
        # 2D  
        B = 1/4
        sum_2d = 0
        for m in range(w):
            cos_term = np.cos(2 * np.pi * m / w)
            denominator = (1 - 2*B*cos_term)**2 - 4*B**2
            if denominator > 1e-15:
                sum_2d += denominator**(-0.5)
        
        Q0_2d = 1 - w/sum_2d if sum_2d > 0 else np.nan
        Q0_2d_values.append(Q0_2d)
        
        print(f"w={w:5d}: 3D={Q0_3d:8.4f}, 2D={Q0_2d:8.4f}")
    
    Q0_3d_values = np.array(Q0_3d_values)
    Q0_2d_values = np.array(Q0_2d_values)
    
    # גרף
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 3D case
    valid_3d = ~np.isnan(Q0_3d_values) & (Q0_3d_values > -1) & (Q0_3d_values < 1)
    ax1.semilogx(w_values[valid_3d], Q0_3d_values[valid_3d], 'bo-', label='3D Discrete')
    ax1.axhline(y=0.3405, color='r', linestyle='--', label='3D Continuous ≈ 0.3405')
    ax1.axhline(y=0.35, color='orange', linestyle=':', label='3D Random Walk ≈ 0.35')
    ax1.set_xlabel('w (log scale)')
    ax1.set_ylabel('Q₀*')
    ax1.set_title('3D Case Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 0.5)
    
    # 2D case  
    valid_2d = ~np.isnan(Q0_2d_values)
    ax2.semilogx(w_values[valid_2d], Q0_2d_values[valid_2d], 'mo-', label='2D Discrete')
    ax2.axhline(y=1.0, color='r', linestyle='--', label='2D Target = 1.0')
    ax2.set_xlabel('w (log scale)')
    ax2.set_ylabel('Q₀*')
    ax2.set_title('2D Case Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.show()

# הרץ ניתוח
w_vals, cont_3d, cont_2d = test_convergence_detailed()
plot_convergence_analysis()

print(f"\nמסקנה:")
print(f"w=100 זה קטן מדי! צריך לפחות w≥1000 כדי לראות התכנסות נכונה")
print(f"הבעיה היא שהקירוב הדיסקרטי זקוק לרזולוציה גבוהה")