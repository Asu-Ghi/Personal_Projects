#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

#define max_array = 256

typedef struct node {
    int val;
    struct node* next;
 } node_t;

// print linked list
void print_list(node_t* head) {
    if (head == NULL) {
        printf("List is empty \n");
    }
    node_t * current = head;

    while (current!=NULL) {
        printf("%d\n", current -> val);
        current = current -> next;
    }
}

// push to end of linked list
void push_end(node_t* head, int val) {
    node_t* current = head;
    while(current -> next != NULL) {
        current = current -> next;
    }
    current -> next = (node_t*)malloc(sizeof(node_t));
    current -> next -> val = val;
    current -> next -> next = NULL;

}

// push to front of linked list
void push_front(node_t ** head, int val) {
    node_t * new_node = (node_t*) malloc(sizeof(node_t));
    new_node -> val = val;
    // derefrence head to assign to new node, clone head essentially
    new_node -> next = *head;
    // derefrence head to assign to new node.
    *head = new_node;
}

// get length of list
int get_length(node_t* head) {
    if (head == NULL) {
        printf("HERE1\n");
        return 0;
    }
    else if (head -> next == NULL) {
        printf("HERE2\n");
        return 1;
    }

    int count = 1;
    node_t* current = head;
    while (current -> next != NULL) {
        count++;
        current = current->next;
    }
    return count;
}

void push_index(node_t **head, int index, int val) {
    // allocate memory for the new node
    node_t *temp = (node_t *)malloc(sizeof(node_t));

    if (temp == NULL) {
        printf("Memory allocation failed\n");
        return;
    }

    temp->val = val;
    
    // check for valid index
    if (index < 0 || index > get_length(*head)) {
        printf("Index %d out of bounds for linked list.\n", index);
        free(temp); // Free the allocated memory
        return;
    }

    // insertion at the beginning
    if (index == 0) {
        temp->next = *head;
        *head = temp;
        return;
    }

    node_t *current = *head;
    // traverse to the node before the insertion point
    for (int i = 0; i < index - 1; i++) {
        current = current->next;
    }

    // insert the new node
    temp->next = current->next;
    current->next = temp;
}

// pop top element from list
node_t* pop_top(node_t ** head) {

    if (*head == NULL) {
        return NULL;
    }

    node_t* temp = *head;
    *head = (*head) -> next;
    temp -> next = NULL;
    return temp;
}

// pop last element from list 
node_t* pop_end(node_t** head) {
    node_t* current = *head;

    while (current -> next -> next != NULL) {
        current = current -> next;
    }

    // get and delete last element
    node_t* temp = current -> next;
    current -> next = NULL;
    return temp;
}

// delete and element
void remove_element(node_t ** head, int val) {
    node_t * current = *head;

    if (*head == NULL) {
        return; // list is empty.
    }

    // if head = element to remove
    if (current -> val == val) {
        node_t * temp = *head;
        *head = current -> next;
        free(temp);
        return;
    }

    // iterate through list
    while (current -> next != NULL ) {
        if (current -> next -> val == val) {
            node_t * temp = current -> next;
            current -> next = current -> next -> next;
            free(temp);
            return;
        }
        current = current -> next;
    }
}

// remove all elements from the list
void delete_list(node_t** head){

    if (*head == NULL) {
        return;
    }

    node_t * current = *head;
    node_t * next_node;

    while (current != NULL) {
        next_node = current -> next;
        free(current);
        current = next_node;
    }

    *head = NULL;
}

// get node from element
node_t* get_node_from_element(node_t* head, int val) {
    node_t * current = head;

    if (head == NULL) {
        printf("Empty List \n");
        return NULL;
    }

    while (current != NULL) {
        if(current -> val == val) {
            return current;
        } 
        current = current -> next;
    }
    printf("Value %d not found \n", val);
    return NULL;
}

int get_index_from_value(node_t* head, int val) {
    int index = 0;
    if (head == NULL) {
        printf("List is empty.\n");
        return -1;
    }

    // if head has the value
    if (head->val == val) {
        return index;
    }

    // iterate through nodes and incriment index.
    node_t * current = head;
    while (current != NULL) {
        if (current->val == val) {
            return index;
        }
        index++;
        current = current -> next;
    }

    // value not found
    printf("Value not in linked list.\n");
    return -1;


}

void test_functions(){
    node_t * head = NULL;
    head = (node_t *) malloc(sizeof(node_t));
    head -> val = 1;
    // check if head exists
    if (head == NULL) {
        printf("Empty list\n");
    }
    
    // test push_front
    push_front(&head, 2);
    push_front(&head, 3);
    print_list(head);
    printf("----------- \n");
    // 3 2 1

    // test push_end
    push_end(head, 4);
    push_end(head, 5);
    print_list(head);
    printf("----------- \n");
    // 3 2 1 4 5

    // test get_length
    printf("Size of %d \n", get_length(head));
    printf("----------- \n");
    // 5

    // test push_index
    push_index(&head, 3, 6);
    print_list(head);
    printf("----------- \n");
    // 3 2 1 6 4 5

    // test pop_top
    pop_top(&head);
    print_list(head);
    printf("----------- \n");
    // 2 1 6 4 5

    // test pop_end
    pop_end(&head);
    print_list(head);
    printf("----------- \n");
    // 2 1 6 4

    // test remove_element
    remove_element(&head, 6);
    print_list(head);
    printf("----------- \n");
    // 2 1 4

    // test get_node_from_element
    node_t * element = get_node_from_element(head, 2);
    printf ("Element value of %d\n", element->val);
    print_list(head);
    printf("----------- \n");
    // 2

    // test get_index_from_value
    int index = get_index_from_value(head, 4);
    printf("Index of %d for the first node with the value %d\n" , index, 4);
    printf("----------- \n");
    // 2
    
    // test delete_list
    delete_list(&head);
    print_list(head);
    printf("----------- \n");
    // list is empty

}

// main
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("command usage %s %s\n", argv[0], "data");
    }
    
    // run all tests for methods.
    test_functions();

    return 0;
}
