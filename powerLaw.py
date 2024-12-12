import numpy as np
import matplotlib.pyplot as plt

def simulate_particle(start_time, end_time, step_size, initial_position, T, direction_distribution=[0.5, 0.5]):
    """
    סימולציית תנועת חלקיק אקראית חד-ממדית.

    ארגומים:
        start_time: זמן ההתחלה של הסימולציה
        end_time: זמן הסיום של הסימולציה
        step_size: גודל הקפיצה של החלקיק בכל צעד
        initial_position: המיקום ההתחלתי של החלקיק
        T: פונקציית זמן ההמתנה
        direction_distribution: התפלגות ההסתברות לכיוון הקפיצה (ברירת מחדל: שווה הסתברויות)

    החזרה:
        רשימת זמנים ורשימת מיקומים מתאימים.
    """

    time = start_time
    position = initial_position
    times = [time]
    positions = [position]

    while time < end_time:
        direction = np.random.choice([-1, 1], p=direction_distribution)
        wait_time = T(time)  # חישוב זמן ההמתנה לפי הפונקציה החדשה
        time += wait_time
        position += direction * step_size
        times.append(time)
        positions.append(position)

    return times, positions

def main():
    # הגדרת פרמטרים
    start_time = 0.1
    end_time = 10
    step_size = 1
    initial_position = 0

    # הגדרת פונקציית זמן המתנה חדשה (מוגדרת היטב עבור t=0)
    T = lambda t: 0.5 * (t + 1)**(-3/2)

    # הגדרת התפלגות כיוון הקפיצה (שווה הסתברויות)
    direction_distribution = [0.5, 0.5]

    # הפעלת הסימולציה
    times, positions = simulate_particle(start_time, end_time, step_size, initial_position, T, direction_distribution)

    # מציאת האינדקסים בהם החלקיק משנה כיוון
    change_indices = np.where(np.diff(np.sign(np.diff(positions))))[0] + 1

    # הצגת גרף
    plt.plot(times, positions)
    plt.scatter(times[change_indices], positions[change_indices], color='red', marker='o')  # הוספת נקודות אדומות בזמני העצירה
    plt.xlabel("זמן")
    plt.ylabel("מיקום")
    plt.title("תנועה אקראית עם סימון זמני עצירה")
    plt.show()

if __name__ == "__main__":
    main()