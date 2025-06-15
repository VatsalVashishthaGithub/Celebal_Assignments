class Node:
    # A node stores the data and a reference to the next node
    def __init__(this, data):
        this.data = data
        this.next = None


class LinkedList:
    # Initialize the head of the linked list
    def __init__(this):
        this.head = None

    def add_node(this, data):
        # Create a new node with the given data
        new_node = Node(data)

        # If the list is empty, the new node becomes the head
        if this.head is None:
            this.head = new_node
            return

        # Otherwise, traverse to the end and append the new node
        current = this.head
        while current.next:
            current = current.next

        current.next = new_node

    def print_list(this):
        # Print all elements of the linked list
        if this.head is None:
            print("The list is empty.")
            return

        current = this.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(this, n):
        # Delete the node at position n (1-based index)
        if this.head is None:
            raise IndexError("Cannot delete from an empty list.")

        if n <= 0:
            raise ValueError("Index must be a positive integer.")

        if n == 1:
            removed = this.head.data
            this.head = this.head.next
            print(f"Deleted node at position 1 with value {removed}")
            return

        current = this.head
        prev = None
        count = 1

        # Traverse the list to find the nth node
        while current and count < n:
            prev = current
            current = current.next
            count += 1

        if current is None:
            raise IndexError("Index out of range.")

        removed = current.data
        prev.next = current.next
        print(f"Deleted node at position {n} with value {removed}")


# ----------------------------
# Testing the LinkedList class
# ----------------------------
if __name__ == "__main__":
    my_list = LinkedList()

    # Add some elements
    my_list.add_node(5)
    my_list.add_node(15)
    my_list.add_node(25)
    my_list.add_node(35)

    print("Initial Linked List:")
    my_list.print_list()

    # Delete the 3rd node
    try:
        my_list.delete_nth_node(3)
    except Exception as error:
        print("Error:", error)

    print("\nList after deleting 3rd node:")
    my_list.print_list()

    # Try deleting a node at an invalid position
    try:
        my_list.delete_nth_node(10)
    except Exception as error:
        print("\nError:", error)

    # Try deleting from an empty list
    empty_list = LinkedList()
    try:
        empty_list.delete_nth_node(1)
    except Exception as error:
        print("\nError:", error)
