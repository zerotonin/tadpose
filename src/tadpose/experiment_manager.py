# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — experiment_manager                                    ║
# ║  « interactive experiment and investigator setup »               ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Console-driven creation of experiment types, investigators      ║
# ║  and series records in the tadpole database.                     ║
# ╚══════════════════════════════════════════════════════════════════╝
from tadpose.database import *
from prettytable import PrettyTable
import os

class ExperimentManager:
    """
    Manages the creation and management of experiments, including managing investigators and
    experiment types.

    Attributes:
        db_handler (DatabaseHandler): The database handler for accessing experiment-related data.
    """

    def __init__(self, db_handler):
        """
        Initializes the ExperimentManager with a database handler.

        Args:
            db_handler (DatabaseHandler): The database handler for accessing experiment-related data.
        """
        self.db_handler = db_handler

    def _clear_screen(self):
        """
        Clears the terminal screen for better readability.
        """
        os.system('cls' if os.name == 'nt' else 'clear')

    def show_investigators(self):
        """
        Retrieves and formats a PrettyTable of all investigators in the database.
        Returns:
            PrettyTable: Table containing the list of investigators.
        """
        with self.db_handler as db:
            investigators = db.get_records(Investigator)
            table = PrettyTable()
            table.field_names = ["ID", "First Name", "Last Name"]
            for investigator in investigators:
                table.add_row([investigator.investigator_id, investigator.first_name, investigator.last_name])
        return table

    def enter_new_investigator(self):
        """
        Interactively creates a new investigator and adds them to the database after user confirmation.
        """
        first_name = input("Enter the first name of the new investigator: ")
        last_name = input("Enter the last name of the new investigator: ")
        if self.confirm_action("Are you sure you want to add this new investigator?"):
            new_investigator = Investigator(first_name=first_name, last_name=last_name)
            with self.db_handler as db:
                db.add_record(new_investigator)
            print("New investigator added successfully.")
        else:
            print("Action canceled.")
        
    def show_experiment_types(self):
        """
        Continuously retrieves and formats a PrettyTable of all experiment types in the database
        until the user decides not to view any more protocols.
        Returns:
            PrettyTable: Table containing the list of experiment types.
        """
        with self.db_handler as db:
            types = db.get_records(ExperimentType)
            table = PrettyTable()
            table.field_names = ["ID", "Short Name", "Long Name"]
            for exp_type in types:
                table.add_row([exp_type.experiment_type_id, exp_type.short_name, exp_type.long_name])
            print(table)
        return table
    
    def view_protocol(self):
        """
        Asks the user for the ID of the experiment type to view the protocol and displays it.
        """
        with self.db_handler as db:
            types = db.get_records(ExperimentType)
            self.show_experiment_types()
            experiment_id = input("Enter the ID of the experiment to view its protocol: ").strip()
            # Find the experiment type with the given ID
            experiment_type = next((et for et in types if str(et.experiment_type_id) == experiment_id), None)
            if experiment_type:
                self._clear_screen()
                self.display_protocol(experiment_type)
                input("\n\nPress any key to return to the menu...")
            else:
                print("Experiment ID not found.")
                return None  # Return None explicitly if not found
            
    def display_protocol(self, experiment_type):
        """
        Displays the protocol of the specified experiment type.
        """
        print(f"Protocol for {experiment_type.short_name}: {experiment_type.protocol}")


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

    def enter_new_experiment_type(self):
        """
        Interactively creates a new experiment type and adds it to the database after user confirmation,
        including the possibility to add up to five attributes by choosing existing ones or creating new ones.
        """
        short_name = input("Enter the short name of the new experiment type: ")
        long_name = input("Enter the long name of the new experiment type: ")
        protocol = input("Enter a description of the protocol of the experiment: ")

        attributes = self.collect_attributes()

        # Display the experiment details and ask for confirmation
        if self.confirm_experiment(short_name, attributes):
            new_type = ExperimentType(
                short_name=short_name, 
                long_name=long_name, 
                protocol=protocol, 
                attributes=attributes
            )
            with self.db_handler as db:
                db.add_record(new_type)
            print("New experiment type added successfully.")
            input("\n\nPress any key to return to the menu...")
        else:
            print("Action canceled.")
            input("\n\nPress any key to return to the menu...")

    def confirm_experiment(self, short_name, attributes):
        """
        Displays experiment details and asks the user for confirmation before proceeding.
        
        Args:
            short_name (str): Short name of the experiment type.
            attributes (list): List of ExperimentTypeAttributes instances selected for this experiment type.
        
        Returns:
            bool: True if the user confirms, False otherwise.
        """
        print("\nYou are about to add a new experiment type with the following details:")
        print(f"Short Name: {short_name}")
        print(f"Attributes Selected ({len(attributes)}): {[attr.name for attr in attributes]}")
        return input("Are you sure you want to proceed? Type 'yes' to confirm: ").strip().lower() == 'yes'

    def collect_attributes(self):
        """
        Collects up to five attributes, allowing the user to choose from existing attributes, create new ones,
        or delete a selected attribute.
        Returns:
            List of ExperimentTypeAttributes instances.
        """
        attributes = []
        while len(attributes) < 5:
            self.clear_screen()
            print(f"Selected attributes ({len(attributes)}/5): {[attr.name for attr in attributes]}\n")
            self.display_existing_attributes()

            print("\nOptions:")
            print("1: Add an existing attribute")
            print("2: Create a new attribute")
            print("3: Delete an existing attribute from selection")
            print("4: Save the current attribute selection and exit")

            choice = input("Enter your choice: ").strip()
            if choice == '1':
                attribute = self.select_existing_attribute()
                if attribute and attribute not in attributes:
                    attributes.append(attribute)
                else:
                    print("This attribute is already selected or not available.")
            elif choice == '2':
                attribute = self.create_new_attribute()
                attributes.append(attribute)
            elif choice == '3':
                if attributes:
                    self.delete_attribute(attributes)
                else:
                    print("No attributes to delete.")
            elif choice == '4':
                break
            else:
                print("Invalid choice. Please select a valid option.")

        return attributes

    def delete_attribute(self, attributes):
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


    def display_existing_attributes(self):
        """
        Displays existing attributes using PrettyTable.
        """
        with self.db_handler as db:
            existing_attributes = db.get_records(ExperimentTypeAttributes)
            if existing_attributes:
                table = PrettyTable()
                table.field_names = ["ID", "Name"]
                for attr in existing_attributes:
                    table.add_row([attr.experiment_type_attribute_id, attr.name])
                print(table)
            else:
                print("No existing attributes found.")

    def clear_screen(self):
        """
        Clears the terminal screen for better readability.
        """
        os.system('cls' if os.name == 'nt' else 'clear')

    def select_existing_attribute(self):
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
                attribute = db.session.query(ExperimentTypeAttributes).filter_by(experiment_type_attribute_id=attribute_id).first()
            if attribute:
                print(f"Attribute {attribute.name} added.")
                return attribute
            else:
                print("No attribute found with the provided ID.")
                return None
        except ValueError:
            print("Please enter a valid numeric ID.")
            return None

    def create_new_attribute(self):
        """
        Prompts user to enter a name for a new attribute and creates it.
        Returns:
            The new ExperimentTypeAttributes instance.
        """
        attribute_name = input("Enter the name for the new attribute: ").strip()
        new_attribute = ExperimentTypeAttributes(name=attribute_name)
        with self.db_handler as db:
            db.session.add(new_attribute)
            db.session.commit()
            # After commit, refresh the object to update its state and avoid it being detached
            db.session.refresh(new_attribute)
        print("New attribute added.")
        return new_attribute


    def select_investigator(self):
        """
        Allows the user to select an investigator by ID from a displayed list.
        Returns:
            int: The ID of the selected investigator, or None if selection is invalid.
        """
        self.show_investigators()
        try:
            investigator_id = int(input("Enter the ID of the investigator you choose: "))
            with self.db_handler as db:
                investigator = db.get_records(Investigator, filters={'investigator_id': investigator_id})
                if investigator:
                    return investigator_id
        except ValueError:
            pass
        print("Invalid ID. Please enter a valid numeric ID.")
        self.select_investigator()
        return None

    def select_experiment_type(self):
        """
        Allows the user to select an experiment type by ID from a displayed list.
        Returns:
            int: The ID of the selected experiment type, or None if selection is invalid.
        """
        self.show_experiment_types()
        try:
            experiment_type_id = int(input("Enter the ID of the experiment type you choose: "))
            with self.db_handler as db:
                experiment_type = db.get_records(ExperimentType, filters={'experiment_type_id': experiment_type_id})
                if experiment_type:
                    return experiment_type_id
        except ValueError:
            pass
        print("Invalid ID. Please enter a valid numeric ID.")
        self.select_experiment_type()
        return None

    def display_selected_options(self, investigator_id, experiment_type_id):
        """
        Displays the names of the selected investigator and experiment type.
        Args:
            investigator_id (int): ID of the selected investigator.
            experiment_type_id (int): ID of the selected experiment type.
        """
        with self.db_handler as db:
            investigator_first_name = db.get_records(Investigator, filters={'investigator_id': investigator_id})[0].first_name if investigator_id else "None"
            investigator_last_name = db.get_records(Investigator, filters={'investigator_id': investigator_id})[0].last_name if investigator_id else "None"
            experiment_type = db.get_records(ExperimentType, filters={'experiment_type_id': experiment_type_id})[0].short_name if experiment_type_id else "None"
            print("Selected investigator: {} {}".format(investigator_first_name, investigator_last_name))
            print("Selected Experiment Type: {}".format(experiment_type))

    def display_tables_side_by_side(self):
        """
        Displays investigators and experiment types side by side for easy comparison and selection.
        Uses padding to ensure both tables are aligned even if they have different numbers of rows.
        """
        self._clear_screen()
        investigator_table = self.show_investigators()
        type_table = self.show_experiment_types()

        # Split tables into lines
        exp_lines = investigator_table.get_string().split('\n')
        type_lines = type_table.get_string().split('\n')
        
        # Calculate the maximum number of lines either table can have
        max_lines = max(len(exp_lines), len(type_lines))

        # Extend both lists to have the same number of lines
        exp_lines.extend([''] * (max_lines - len(exp_lines)))
        type_lines.extend([''] * (max_lines - len(type_lines)))

        # Format the lines with adequate spacing
        combined_lines = []
        for exp_line, type_line in zip(exp_lines, type_lines):
            # Ensure both lines have the same vertical alignment
            formatted_line = f"{exp_line.ljust(50)}{type_line}"
            combined_lines.append(formatted_line)

        # Print the combined lines for side-by-side display
        for line in combined_lines:
            print(line)

    def manage_experiments(self):
        """
        Main method to start the experiment management process. Allows the user to add, select
        investigators and experiment types, showing both tables side by side for easy reference.
        Returns a dictionary with tions(experimenthe selected investigator and experiment type IDs.
        """
        investigator_id = None
        experiment_type_id = None
        
        while True:
            self.display_tables_side_by_side()
            self.display_selected_options(investigator_id, experiment_type_id)

            print("\nChoose an option:")
            print("1. Add investigator")
            print("2. Add Experiment Type")
            print("3. Select investigator")
            print("4. Select Experiment Type")
            print("5. View Protocols for existing Experiments")
            print("6. Exit")
            choice = input("Enter your choice: ")

            if choice == '1':
                self.enter_new_investigator()
            elif choice == '2':
                self.enter_new_experiment_type()
            elif choice == '3':
                investigator_id = self.select_investigator()
            elif choice == '4':
                experiment_type_id = self.select_experiment_type()
            elif choice == '5':
                experiment_type_id = self.view_protocol()
            elif choice == '6':
                print("entered 6")
                if investigator_id and experiment_type_id:
                    #input("\n\nAll entered correctly Press any key to return to the menu...")
                    return {'investigator_id': investigator_id, 'experiment_type_id': experiment_type_id}
                else:
                    print("Please ensure both investigator and experiment type are selected before exiting.")
                    input("\n\nPress any key to return to the menu...")
            else:
                input("Invalid choice. Please select a valid option.\nPress any Key to continue")

