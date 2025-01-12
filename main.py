import WTD

def main():
    
    while True:

        print("\nMenu:")
        print("1. View Single Random Walk")
        print("2. View Histogram of Final Positions")
        print("3. View log scale of Final Positions")
        print("4. comp wait time func to transform")
        print("9. Exit")
        choice = input("Enter your choice (1-4) or 9 to EXIT: ")

        if choice == '1':
            sim_time = int(input("Enter simulation time: "))
            prob_right = float(input("Enter probability of moving right: "))
            WTD.view_single_RW(sim_time, prob_right)

        elif choice == '2':
            num_sims = int(input("Enter number of simulations: "))
            sim_time = int(input("Enter simulation time: "))
            prob_right = float(input("Enter probability of moving right: "))
            WTD.view_hist(num_sims, sim_time, prob_right)

        elif choice =='3':
            num_sims = int(input("Enter number of simulations: "))
            sim_time = int(input("Enter simulation time: "))
            prob_right = float(input("Enter probability of moving right: "))
            WTD.view_logScale(num_sims, sim_time, prob_right)

        elif choice == '4':
            WTD.comp_timeFunc_toTransform()

        elif choice == '9':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 5 or 9 to EXIT.")

if __name__ == "__main__":
    main()