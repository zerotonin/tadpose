import os
from prettytable import PrettyTable
from sqlalchemy.orm import sessionmaker
from tadpose.database import *
from datetime import datetime
class PlateManager:
    def __init__(self, db_handler):
        self.db_handler = db_handler
        self.plate_well_state, self.plate_tadpole_state = self.initialise_plate_state()
        self.well_types = []
        self.tadpole_groups=[]
        self.well_type_ids = []
        self.tadpole_group_ids = []
        

    def initialise_plate_state(self):
        """
        Initializes the plate state with lists for wells and tadpoles, each having 24 entries set to None.
        """
        # One list for wells and one for tadpoles, each with 24 slots initially set to None
        plate_well_state = [None] * 24
        plate_tadpole_state = [None] * 24
        return plate_well_state, plate_tadpole_state
    
    def clear_screen(self):
        """
        Clears the terminal screen for better readability.
        """
        os.system('cls' if os.name == 'nt' else 'clear')


    def visualize_plate(self):
        """
        Visualizes the current state of the plate using PrettyTable with cells labeled from 1 to 24,
        arranged in 6 columns and 4 rows, with lines between rows for clarity and no headers.
        """
        # Setup the table
        pt = PrettyTable()
        pt.header = False  # No header needed
        pt.border = True  # Enable the outer border for clarity
        pt.horizontal_char = '-'  # Horizontal separator character
        pt.vertical_char = '|'  # Vertical separator character
        pt.junction_char = '+'  # Junction character
        pt.align = 'l'  # Left align
        pt.padding_width = 1  # No padding around content

        # Formatting the table output into 6 columns and 4 rows
        rows = []
        for row in range(4):
            row_data = []
            for col in range(6):
                index = row * 6 + col
                well_id = "None" if self.plate_well_state[index] is None else self.plate_well_state[index]
                tadpole_id = "None" if self.plate_tadpole_state[index] is None else self.plate_tadpole_state[index]
                cell_data = f"W: {well_id}\nT: {tadpole_id}"
                row_data.append(cell_data)
            rows.append(row_data)

        # Add all rows to the table, add horizontal lines only between different sets of rows
        for i, row_data in enumerate(rows):
            pt.add_row(row_data)
            if i < len(rows) - 1:  # Add horizontal lines between rows except for the last row
                pt.add_row(['------'] * 6)  # Use a placeholder that fits the expected width of the cell
        print("Position in Plate of each Well ID (W) and Tadole ID (T)")
        print(pt.get_string())
        print("\n")

    def visualize_plate_wells_only(self):
        """
        Visualizes the current state of the plate using PrettyTable with cells labeled from 1 to 24,
        arranged in 6 columns and 4 rows, with lines between rows for clarity and no headers.
        """
        # Setup the table
        pt = PrettyTable()
        pt.header = False  # No header needed
        pt.border = True  # Enable the outer border for clarity
        pt.horizontal_char = '-'  # Horizontal separator character
        pt.vertical_char = '|'  # Vertical separator character
        pt.junction_char = '+'  # Junction character
        pt.align = 'l'  # Left align
        pt.padding_width = 1  # No padding around content

        # Formatting the table output into 6 columns and 4 rows
        rows = []
        for row in range(4):
            row_data = []
            for col in range(6):
                index = row * 6 + col
                well_id = "None" if self.plate_well_state[index] is None else self.plate_well_state[index]
                cell_data = f"{well_id}"
                row_data.append(cell_data)
            rows.append(row_data)

        # Add all rows to the table
        for row_data in rows:
            pt.add_row(row_data)

        print("Current state of the plate (Well IDs)")
        print(pt.get_string())
        print("\n")
        
    def visualize_plate_tadpoles_only(self):
        """
        Visualizes the current state of the plate using PrettyTable with cells labeled from 1 to 24,
        arranged in 6 columns and 4 rows, with lines between rows for clarity and no headers.
        """
        # Setup the table
        pt = PrettyTable()
        pt.header = False  # No header needed
        pt.border = True  # Enable the outer border for clarity
        pt.horizontal_char = '-'  # Horizontal separator character
        pt.vertical_char = '|'  # Vertical separator character
        pt.junction_char = '+'  # Junction character
        pt.align = 'l'  # Left align
        pt.padding_width = 1  # No padding around content

        # Formatting the table output into 6 columns and 4 rows
        rows = []
        for row in range(4):
            row_data = []
            for col in range(6):
                index = row * 6 + col
                tadpole_id = "None" if self.plate_tadpole_state[index] is None else self.plate_tadpole_state[index]
                cell_data = f"{tadpole_id}"
                row_data.append(cell_data)
            rows.append(row_data)

        # Add all rows to the table
        for row_data in rows:
            pt.add_row(row_data)

        print("Current state of the plate (Tadpole IDs)")
        print(pt.get_string())
        print("\n")
        
    def display_selected_well_types(self, well_types):
        # Ensure the screen is clear before displaying the information

        # Check if there are well types to display
        if not well_types:
            print("No well types selected.")
            return

        # Create a PrettyTable object with column headers
        table = PrettyTable()
        table.field_names = ["ID", "Name"]
        table.align["ID"] = "l"  # Align the IDs left
        table.align["Name"] = "l"  # Align the names left

        # Iterate over each well type and add a row to the table
        for well_type in well_types:
            table.add_row([well_type.well_type_id, well_type.name])
            
        # Print the table
        return(table)
    


    def manage_plate(self):
        while True:
            self.clear_screen()
            self.visualize_plate()
            print(self.display_selected_well_types(self.well_types))
            print(self.display_selected_tadpole_groups(self.tadpole_groups))
            print("Plate Manager Main Menu")
            print()
            print("1. Edit Wells")
            print("2. Edit Tadpoles")
            print("3. Save and Exit")
            choice = input("Enter your choice: ")
            if choice == '1':
                self.manage_wells()
            elif choice == '2':
                self.manage_tadpoles()
            elif choice == '3':
                print("Exiting Plate Manager.")
                return {'well_type_ids': self.plate_well_state, 'tadpole_type_ids': self.plate_tadpole_state}
            else:
                print("Invalid choice. Please select a valid option.")


    def manage_wells(self):
        self.clear_screen
        well_types=self.well_types
        well_types = self.manage_well_types(well_types)
        while True:
            self.clear_screen()
            print("Current Well Type Selection")
            print(self.display_selected_well_types(well_types))
            print("Current Well Layout\n")
            self.visualize_plate_wells_only()
            print("Manage Wells")
            print("1. Edit Well types")
            print("2. Assign Well Types to Positions")
            print("3. Save Selection and Return to Main Menu")
            choice = input("Enter your choice: ")
            if choice == '2':
                choice= self.assign_well_types_to_positions(well_types)
            if choice == '1':
                self.manage_well_types(well_types)
            if choice == '3':
                break
            else:
                print("Invalid choice. Please select a valid option.")
                

                
    def assign_well_types_to_positions(self, well_types):
        self.display_selected_well_types(well_types)

        while True:
            input_ids = input("Enter well IDs to edit (e.g., 1, 2-4, 9-11): ")
            indices = []
            for part in input_ids.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    indices.extend(range(start, end + 1))
                else:
                    indices.append(int(part))

            valid_well_type_ids = [wt.well_type_id for wt in well_types]
            well_type_id = None

            while True:
                try:
                    well_type_id = int(input("Enter well type ID for these positions: "))
                    if well_type_id in valid_well_type_ids:
                        break
                    else:
                        print(f"Invalid well type ID. Valid types are: {', '.join(map(str, valid_well_type_ids))}")
                except ValueError:
                    print("Please enter a valid numeric ID.")

            for index in indices:
                if 1 <= index <= 24:
                    self.plate_well_state[index - 1] = well_type_id

            self.clear_screen()
            self.display_selected_well_types(well_types)
            self.visualize_plate_wells_only()

            while True:
                print(self.display_selected_well_types(well_types))
                print("1. Position more well types in plate")
                print("2. Edit Well Type selection")
                print("3. Save and exit")
                choice = input("Select an option: ")

                if choice == '1':
                    break
                elif choice == '2':
                    return 1
                elif choice == '3':
                    if None in self.plate_well_state:
                        print("Not all wells have been given a type. Are you sure you want to proceed with this selection? (yes/no)")
                        confirmation = input()
                        if confirmation.lower() == 'yes':
                            return 3
                        else:
                            continue
                    else:
                        return 3
                else:
                    print("Invalid choice. Please select a valid option.")


    def confirm_action(self, prompt_message):
        """
        Asks the user for confirmation before proceeding with an action.

        Args:
            prompt_message (str): The message to display to the user asking for confirmation.

        Returns:
            bool: True if the user confirms the action, False otherwise.
        """
        confirmation = input(prompt_message + " Type 'yes' to confirm: ").strip().lower()
        return confirmation == 'yes'
    
    
    def manage_well_types(self, well_types):

        while True:
            self.clear_screen()
            print(self.display_existing_well_types())
            print(f"Selected well types ({len(well_types)}): {[attr.name for attr in well_types]}\n")
            #self.display_existing_well_types()

            print("\nOptions:")
            print("1: Select an existing well type")
            print("2: Create a new well type")
            print("3: Delete an existing well type from selection")
            print("4: Save the current well type selection and return to well plate menu")

            choice = input("Enter your choice: ").strip()
            if choice == '1':
                well_type = self.select_existing_well_type()
                if well_type==None:
                    continue
                if well_type and well_type not in well_types:
                    well_types.append(well_type)
                else:
                    print("This well type is already selected or not available.")
            elif choice == '2':
                well_type = self.enter_new_well_type()
                if well_type==None: 
                    continue
                well_types.append(well_type)
            elif choice == '3':
                if well_types:
                    self.delete_well_type(well_types)
                else:
                    print("No well types to delete.")
            elif choice == '4':
                if len(well_types)<1:
                    input("Cannot exit as no well type is selected.\n Please select at least one well type.\n Press Any key to return to the Well type selection menu...")
                return(well_types)
            else:
                print("Invalid choice. Please select a valid option.")
                
    def delete_well_type(self, well_types):
        """
        Allows the user to delete a well type from the selected list by specifying its index.
        """
        for idx, attr in enumerate(well_types, start=1):
            print(f"{idx}: {attr.name}")
        try:
            index = int(input("Enter the number of the well type to delete: ")) - 1
            if 0 <= index < len(well_types):
                del well_types[index]
                print("Attribute deleted successfully.")
            else:
                print("Invalid index. Please enter a valid number.")
        except ValueError:
            print("Please enter a valid number.")




    def select_existing_well_type(self):
        """
        Allows the user to select an existing attribute by entering its ID.
        Returns:
            The selected ExperimentTypeAttributes instance or None if not found.
        """
        print(self.display_existing_well_types())
        well_id = input("Enter the ID of the well type to Select: ").strip()
        try:
            well_id = int(well_id)
            with self.db_handler as db:
                # Directly use first() on the query object before fetching or listing all
                well_type = db.session.query(WellType).filter_by(well_type_id=well_id).first()
            if well_type:
                print(f"Well Type {well_type.name} selected.")
                return well_type
            else:
                print("No well type found with the provided ID.")
                return None
        except ValueError:
            print("Please enter a valid numeric ID.")
            return None
        
    def display_existing_well_type_attributes(self):
        """
        Continuously retrieves and formats a PrettyTable of all well types in the database
        until the user decides not to view any more protocols.
        Returns:
            PrettyTable: Table containing the list of experiment types.
        """
        with self.db_handler as db:
            types = db.get_records(WellTypeAttributes)
            table = PrettyTable()
            table.field_names = ["ID", "Name"]
            for well_attribute_type in types:
                table.add_row([well_attribute_type.well_type_attribute_id, well_attribute_type.name])
            #print(table)
        return table
    


    def display_existing_well_types(self):
        """
        Continuously retrieves and formats a PrettyTable of all well types in the database
        until the user decides not to view any more protocols.
        Returns:
            PrettyTable: Table containing the list of experiment types.
        """
        print("entered display method")
        with self.db_handler as db:
            types = db.get_records(WellType)
            table = PrettyTable()
            table.field_names = ["ID", "Name", "Description"]
            for well_type in types:
                table.add_row([well_type.well_type_id, well_type.name, well_type.description])
        return table

    def enter_new_well_type(self):
        """
        Interactively creates a new experiment type and adds it to the database after user confirmation,
        including the possibility to add up to five attributes by choosing existing ones or creating new ones.
        """
        name = input("Enter a short name for the well type (<10 characters): ")
        description = input("Enter a short description of what is in the well (eg. 3ml 0.001 mM PTZ)")
        attributes = self.collect_well_attributes()

        # Display the experiment details and ask for confirmation
        if self.confirm_well_type(name, attributes):
            new_type = WellType(
                name=name, 
                description=description, 
                attributes=attributes
            )
            with self.db_handler as db:
                db.add_record(new_type)
            print("New well type added successfully.")
            with self.db_handler as db:
                well_type = db.session.query(WellType).filter_by(name=name).first()
            if well_type:
                print(f"Well Type {well_type.name} added.")
                return well_type
            input("\n\nPress any key to return to the menu...")
        else:
            print("Action canceled.")
            input("\n\nPress any key to return to the menu...")


    def collect_well_attributes(self):
        """
        Collects up to five attributes, allowing the user to choose from existing attributes, create new ones,
        or delete a selected attribute.
        Returns:
            List of ExperimentTypeAttributes instances.
        """
        attributes = []
        while True:
            self.clear_screen()
            print("Select Attributes related to the well:")
            print(f"Selected attributes ({len(attributes)}): {[attr.name for attr in attributes]}\n")
            print(self.display_existing_well_type_attributes())

            print("\nOptions:")
            print("1: Select an existing well attribute")
            print("2: Create a new well attribute")
            print("3: Delete an existing well attribute from selection")
            print("4: Save the current attribute selection and exit")

            choice = input("Enter your choice: ").strip()
            if choice == '1':
                attribute = self.select_existing_well_attribute()
                if attribute and attribute not in attributes:
                    attributes.append(attribute)
                else:
                    print("This attribute is already selected or not available.")
            elif choice == '2':
                attribute = self.create_new_well_attribute()
                attributes.append(attribute)
            elif choice == '3':
                if attributes:
                    self.delete_well_attribute(attributes)
                else:
                    print("No attributes to delete.")
            elif choice == '4':
                break
            else:
                print("Invalid choice. Please select a valid option.")

        return attributes

    def select_existing_well_attribute(self):
        """
        Allows the user to select an existing attribute by entering its ID.
        Returns:
            The selected ExperimentTypeAttributes instance or None if not found.
        """
        attribute_id = input("Enter the ID of the attribute to add: ").strip()
        try:
            attribute_id = int(attribute_id)
            with self.db_handler as db:
                # Directly use first() on the query object before fetching or listing all
                attribute = db.session.query(WellTypeAttributes).filter_by(well_type_attribute_id=attribute_id).first()
            if attribute:
                print(f"Attribute {attribute.name} added.")
                return attribute
            else:
                print("No attribute found with the provided ID.")
                return None
        except ValueError:
            print("Please enter a valid numeric ID.")
            return None
        


    def create_new_well_attribute(self):
        """
        Prompts user to enter a name for a new attribute and creates it.
        Returns:
            The new ExperimentTypeAttributes instance.
        """
        attribute_name = input("Enter the name for the new attribute: ").strip()
        new_attribute = WellTypeAttributes(name=attribute_name)
        with self.db_handler as db:
            db.session.add(new_attribute)
            db.session.commit()
            # After commit, refresh the object to update its state and avoid it being detached
            db.session.refresh(new_attribute)
        print("New attribute added.")
        return new_attribute
    
    def delete_well_attribute(self, attributes):
        """
        Allows the user to delete an attribute from the selected list by specifying its index.
        """
        for idx, attr in enumerate(attributes, start=1):
            print(f"{idx}: {attr.name}")
        try:
            index = int(input("Enter the number of the attribute to delete: ")) - 1
            if 0 <= index < len(attributes):
                del attributes[index]
                print("Attribute deleted successfully.")
            else:
                print("Invalid index. Please enter a valid number.")
        except ValueError:
            print("Please enter a valid number.")



    def confirm_well_type(self, name, attributes):
        """
        Displays experiment details and asks the user for confirmation before proceeding.
        
        Args:
            short_name (str): Short name of the experiment type.
            attributes (list): List of ExperimentTypeAttributes instances selected for this experiment type.
        
        Returns:
            bool: True if the user confirms, False otherwise.
        """
        print("\nYou are about to add a new well type with the following details:")
        print(f"Short Name: {name}")
        print(f"Attributes Selected ({len(attributes)}): {[attr.name for attr in attributes]}")
        return input("Are you sure you want to proceed? Type 'yes' to confirm: ").strip().lower() == 'yes'





#toedit tadpoles




    def manage_tadpoles(self):
        self.clear_screen
        tadpole_groups=self.tadpole_groups
        tadpole_groups = self.manage_tadpole_groups(tadpole_groups)
        while True:
            self.clear_screen()
            print("Current Tadpole group Selection")
            print(self.display_selected_tadpole_groups(tadpole_groups))
            print("Current Tadpole Layout\n")
            self.visualize_plate_tadpoles_only()
            print("Manage Tadpoles")
            print("1. Edit Tadpole groups")
            print("2. Assign Tadpole groups to Positions")
            print("3. Save Selection and Return to Main Menu")
            choice = input("Enter your choice: ")
            if choice == '2':
                choice= self.assign_tadpole_groups_to_positions(tadpole_groups)
            if choice == '1':
                self.manage_tadpole_groups(tadpole_groups)
            if choice == '3':
                break
            else:
                print("Invalid choice. Please select a valid option.")
                

                
    def assign_tadpole_groups_to_positions(self, tadpole_groups):
        self.display_selected_tadpole_groups(tadpole_groups)

        while True:
            input_ids = input("Enter tadpole IDs to edit (e.g., 1, 2-4, 9-11): ")
            indices = []
            for part in input_ids.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    indices.extend(range(start, end + 1))
                else:
                    indices.append(int(part))

            valid_tadpole_group_ids = [wt.tadpole_group_id for wt in tadpole_groups]
            tadpole_group_id = None

            while True:
                try:
                    tadpole_group_id = int(input("Enter tadpole group ID for these positions: "))
                    if tadpole_group_id in valid_tadpole_group_ids:
                        break
                    else:
                        print(f"Invalid tadpole group ID. Valid groups are: {', '.join(map(str, valid_tadpole_group_ids))}")
                except ValueError:
                    print("Please enter a valid numeric ID.")

            for index in indices:
                if 1 <= index <= 24:
                    self.plate_tadpole_state[index - 1] = tadpole_group_id

            self.clear_screen()
            self.display_selected_tadpole_groups(tadpole_groups)
            self.visualize_plate_tadpoles_only()

            while True:
                self.clear_screen()
                print(self.display_selected_tadpole_groups(tadpole_groups))
                self.visualize_plate_tadpoles_only()
                print("1. Position more tadpole groups in plate")
                print("2. Edit tadpole group selection")
                print("3. Save and exit")
                choice = input("Select an option: ")

                if choice == '1':
                    break
                elif choice == '2':
                    return 1
                elif choice == '3':
                    if None in self.plate_tadpole_state:
                        print("Not all tadpoles have been given a group. Are you sure you want to proceed with this selection? (yes/no)")
                        confirmation = input()
                        if confirmation.lower() == 'yes':
                            return 3
                        else:
                            continue
                    else:
                        return 3
                else:
                    print("Invalid choice. Please select a valid option.")


    
    def manage_tadpole_groups(self,tadpole_groups):
        while True:
            self.clear_screen()
            print(self.display_existing_tadpole_groups())
            print(f"Selected tadpole group IDs: ({len(tadpole_groups)}): {[gp.tadpole_group_id for gp in tadpole_groups]}\n")
            #self.display_existing_tadpole_groups()

            print("\nOptions:")
            print("1: Select an existing tadpole group")
            print("2: Create a new tadpole group")
            print("3: Delete an existing tadpole group from selection")
            print("4: Save the current tadpole group selection and return to tadpole plate menu")

            choice = input("Enter your choice: ").strip()
            if choice == '1':
                tadpole_group = self.select_existing_tadpole_group()
                if tadpole_group==None:
                    continue
                if tadpole_group and tadpole_group not in tadpole_groups:
                    tadpole_groups.append(tadpole_group)
                else:
                    print("This tadpole group is already selected or not available.")
            elif choice == '2':
                tadpole_group = self.enter_new_tadpole_group()
                if tadpole_group==None: 
                    continue
                tadpole_groups.append(tadpole_group)
            elif choice == '3':
                if tadpole_groups:
                    self.delete_tadpole_group(tadpole_groups)
                else:
                    print("No tadpole groups to delete.")
            elif choice == '4':
                if len(tadpole_groups)<1:
                    input("Cannot exit as no tadpole group is selected.\n Please select at least one tadpole group.\n Press Any key to return to the tadpole group selection menu...")
                return(tadpole_groups)
            else:
                print("Invalid choice. Please select a valid option.")
                
    def delete_tadpole_group(self, tadpole_groups):
        """
        Allows the user to delete a tadpole group from the selected list by specifying its index.
        """
        for idx, attr in enumerate(tadpole_groups, start=1):
            print(f"{idx}: {attr.tadpole_group_id}")
        try:
            index = int(input("Enter the number of the tadpole group to delete: ")) - 1
            if 0 <= index < len(tadpole_groups):
                del tadpole_groups[index]
                print("Attribute deleted successfully.")
            else:
                print("Invalid index. Please enter a valid number.")
        except ValueError:
            print("Please enter a valid number.")




    def select_existing_tadpole_group(self):
        """
        Allows the user to select an existing attribute by entering its ID.
        Returns:
            The selected ExperimentgroupAttributes instance or None if not found.
        """
        print(self.display_existing_tadpole_groups())
        tadpole_group_id = input("Enter the ID of the tadpole group to Select ").strip()
        try:
            tadpole_group_id = int(tadpole_group_id)
            with self.db_handler as db:
                # Directly use first() on the query object before fetching or listing all
                tadpole_group = db.session.query(TadpoleGroup).filter_by(tadpole_group_id=tadpole_group_id).first()
            if tadpole_group:
                print(f"tadpole group {tadpole_group.tadpole_group_id} added.")
                return tadpole_group
            else:
                print("No tadpole group found with the provided ID.")
                return None
        except ValueError:
            print("Please enter a valid numeric ID.")
            return None
        

    


    def display_existing_tadpole_groups(self):
        """
        Retrieves and formats a PrettyTable of all tadpole groups in the database
        with additional mother's unique identifier, displaying until user decides to stop.
        Returns:
            PrettyTable: Table containing the list of experiment groups.
        """
        print("Entered display method")
        with self.db_handler as db:
            # Perform a joined query to get the necessary data
            groups = db.session.query(
                TadpoleGroup.tadpole_group_id,
                Frog.female_identifier,
                TadpoleGroup.fertilisation_date,
                TadpoleGroup.transgene
            ).join(Frog, TadpoleGroup.mother_id == Frog.frog_id).all()

            table = PrettyTable()
            table.field_names = ["ID", "Mother Unique ID", "Fertilisation Date", "Transgene"]
            for tadpole_group in groups:
                table.add_row([
                    tadpole_group.tadpole_group_id,
                    tadpole_group.female_identifier,
                    tadpole_group.fertilisation_date,
                    tadpole_group.transgene
                ])
            return table
        
    def display_selected_tadpole_groups(self, tadpole_groups):
        """
        Displays selected tadpole groups with their detailed information including mother's unique ID.

        Args:
            tadpole_groups (list): A list of TadpoleGroup objects to display.

        Returns:
            None: Outputs directly to console using PrettyTable.
        """
        # Check if there are tadpole groups to display
        if not tadpole_groups:
            print("No tadpole groups selected.")
            return

        # Create a PrettyTable object with column headers
        table = PrettyTable()
        table.field_names = ["ID", "Mother Unique ID", "Fertilisation Date", "Transgene"]

        # Retrieve information for each tadpole group
        for tadpole_group in tadpole_groups:
            mother = self.db_handler.session.query(Frog).filter_by(frog_id=tadpole_group.mother_id).first()
            mother_id = mother.female_identifier if mother else "Unknown"
            table.add_row([
                tadpole_group.tadpole_group_id,
                mother_id,
                tadpole_group.fertilisation_date.strftime("%Y-%m-%d"),
                tadpole_group.transgene
            ])
    
        return(table)


    def enter_valid_date(self):
        """
        Prompts the user for a date until a valid format (DD-MM-YYYY) is entered.
        Returns:
            datetime.date: The valid entered date.
        """
        while True:
            date_input = input("Enter the fertilisation date (DD-MM-YYYY): ")
            try:
                # Attempt to parse the date from the input string
                valid_date = datetime.strptime(date_input, "%d-%m-%Y").date()
                return valid_date
            except ValueError:
                print("Invalid date format. Please use DD-MM-YYYY format.")



    def get_valid_development_stage(self):
        """
        Prompts the user for a development stage and validates that it is an integer.
        """
        while True:
            try:
                development_stage = int(input("Enter the development stage: "))
                return development_stage
            except ValueError:
                print("Invalid input. Please enter a valid integer for the development stage.")
    def enter_valid_time(self):
        """
        Prompts the user for a time and validates that it is in a correct format.
        Returns:
            datetime.time: The valid entered time.
        """
        while True:
            time_input = input("Enter the fertilisation time (HH:MM): ")
            try:
                valid_time = datetime.strptime(time_input, "%H:%M").time()
                return valid_time
            except ValueError:
                print("Invalid time format. Please use HH:MM format.")
        
    def enter_new_tadpole_group(self):
        mother = self.assign_tadpole_mother()
        if not mother:
            print("No mother assigned. Exiting the tadpole group creation process.")
            return

        fertilisation_date = self.enter_valid_date()
        fertilisation_time = self.enter_valid_time() 
        fertilisation_datetime = datetime.combine(fertilisation_date, fertilisation_time)  # Combine date and time

        development_stage = self.get_valid_development_stage()
        seq_folder = input("Enter the sequencing folder path: ")
        transgene = input("Enter the transgene: ")

        if self.confirm_tadpole_group(mother, fertilisation_datetime, development_stage, seq_folder, transgene):
            new_group = TadpoleGroup(
                mother_id=mother.frog_id,
                fertilisation_date=fertilisation_datetime,
                development_stage=development_stage,
                seq_folder=seq_folder,
                transgene=transgene
            )
            with self.db_handler as db:
                db.add_record(new_group)
                db.session.commit()  # Ensure the session commits if not automatically committing
                # Fetch the new group freshly from the database
                new_group = db.session.query(TadpoleGroup).filter_by(
                    mother_id=mother.frog_id,
                    fertilisation_date=fertilisation_date,
                    seq_folder=seq_folder,
                    transgene=transgene
                ).order_by(TadpoleGroup.tadpole_group_id.desc()).first()  # Assuming id is auto-increment
            print("New tadpole group added successfully.")
            return new_group
        else:
            print("Action canceled.")
        input("\n\nPress any key to return to the menu...")

    def assign_tadpole_mother(self):
        """
        Collects up to five mothers, allowing the user to choose from existing mothers, create new ones,
        or delete a selected mother.
        Returns:
            List of Experimentgroupmothers instances.
        """
        mother = None
        while True:
            self.clear_screen()
            print("Existing Tadpole Group Mothers")
            print(self.display_existing_tadpole_mothers())
            if mother==None:
                print("No Mother Currently Selected")
            else:
                print("Current Selection ID:"+ str(mother.frog_id))
            print("\nOptions:")
            print("1: Select existing tadpole mother")
            print("2: Create a new tadpole mother")
            print("3: Save the current mother selection and exit")

            choice = input("Enter your choice: ").strip()
            if choice == '1':
                mother = self.select_existing_tadpole_mother()

            elif choice == '2':
                mother = self.create_new_tadpole_mother()
            elif choice == '3':
                break
            else:
                print("Invalid choice. Please select a valid option.")

        return mother


    def display_existing_tadpole_mothers(self):
        """
        Continuously retrieves and formats a PrettyTable of all mother types in the database
        until the user decides not to view any more protocols.
        Returns:
            PrettyTable: Table containing the list of experiment types.
        """
        with self.db_handler as db:
            frogs = db.get_records(Frog)
            table = PrettyTable()
            table.field_names = ["ID", "Unique Identifier", "Background Strain"]
            for frog in frogs:
                table.add_row([frog.frog_id,frog.female_identifier,frog.background_strain])
            #print(table)
        return table
    


    def select_existing_tadpole_mother(self):
        """
        Allows the user to select an existing mother by entering its ID.
        Returns:
            The selected Experimentgroupmothers instance or None if not found.
        """
        mother_id = input("Enter the ID of the mother to select: ").strip()
        try:
            mother_id = int(mother_id)
            with self.db_handler as db:
                # Directly use first() on the query object before fetching or listing all
                mother = db.session.query(Frog).filter_by(frog_id=mother_id).first()
            if mother:
                print(f"mother {mother.female_identifier} selected.")
                return mother
            else:
                print("No mother found with the provided ID.")
                return None
        except ValueError:
            print("Please enter a valid numeric ID.")
            return None
        


    def create_new_tadpole_mother(self):
        """
        Prompts user to enter a name for a new mother and creates it.
        Returns:
            The new Experimentgroupmothers instance.
        """
        female_tank = input("Enter the female tank for the new mother: ").strip()
        female_identifier= input("Enter the female identifier for the new mother: ").strip()
        background_strain= input("Enter the background strain for the new mother: ").strip()
        new_mother = Frog(female_tank=female_tank, female_identifier=female_identifier, background_strain=background_strain)
        with self.db_handler as db:
            db.session.add(new_mother)
            db.session.commit()
            # After commit, refresh the object to update its state and avoid it being detached
            db.session.refresh(new_mother)
        print("New mother added.")
        return new_mother
    

    def confirm_tadpole_group(self, mother, fertilisation_date, development_stage, seq_folder, transgene):
        """
        Displays experiment details and asks the user for confirmation before proceeding.
        
        Returns:
            bool: True if the user confirms, False otherwise.
        """
        print("\nYou are about to add a new tadpole group with the following details:")
        print(f"Mother's Identifier: {mother.female_identifier}")
        print(f"Fertilisation Date: {fertilisation_date}")
        print(f"Development Stage: {development_stage}")
        print(f"Sequencing Folder Path: {seq_folder}")
        print(f"Transgene: {transgene}")
        return input("Are you sure you want to proceed? (yes/no): ").strip().lower() == 'yes'
