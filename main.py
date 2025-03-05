import WTD

def main():
    
    while True:

        print("\nMenu:")
        print("1. View Single Random Walk")
        print("2. View Histogram of Final Positions")
        print("3. View log scale of Final Positions")
        print("4. comp wait time func to transform")
        print("5. To calc the second moment")

        print("9. Exit")

        choice = input("Enter your choice (1-4) or 9 to EXIT: ")

        if choice == '1':
            #sim_time = int(input("Enter simulation time: "))
            #prob_right = float(input("Enter probability of moving right: "))
            WTD.view_single_RW(sim_time = 15_000, prob_right = 0.5)

        elif choice == '2':
            #num_sims = int(input("Enter number of simulations: "))
            #sim_time = int(input("Enter simulation time: "))
            #prob_right = float(input("Enter probability of moving right: "))
            WTD.view_hist(num_sims = 200_000, sim_time = 10_000, prob_right = 0.5)

        elif choice =='3':
            #num_sims = int(input("Enter number of simulations: "))
            #sim_time = int(input("Enter simulation time: "))
            #prob_right = float(input("Enter probability of moving right: "))
            WTD.view_logScale(num_sims = 200_000, sim_time = 15_000, prob_right = 0.5)

        elif choice == '4':
            WTD.comp_timeFunc_toTransform()

        elif choice == '5':
            #sim_time = int(input("Enter simulation time: "))
            #prob_right = float(input("Enter probability of moving right: "))
            positions,times = WTD.RW_sim(sim_time=500, prob_right=0.5)
            print(f"the positions are: {positions}")
            # Calculate the second moment of the positions
            second_moment = WTD.calculate_second_moment(positions)
            print(f"Second moment of positions: {second_moment}")
            

        elif choice == '9':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 5 or 9 to EXIT.")

if __name__ == "__main__":
    main()