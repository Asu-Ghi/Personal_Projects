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
    node_t * current = *head;
    node_t * next_node;

    while (current != NULL) {
        next_node = current -> next;
        free(current);
        current = next_node;
    }

    *head == NULL;
    return;
}




// main
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("command usage %s %s\n", argv[0], "data");
    }
    // create head
    node_t * head = NULL;
    head = (node_t *) malloc(sizeof(node_t));
    head -> val = 1;
    // check if head exists
    if (head == NULL) {
        return 1;
    }

    // add nodes
    push_end(head, 2);
    push_end(head, 3);
    push_end(head, 4);
    push_end(head, 5);

    push_front(&head, 100);
    push_front(&head, 90);
    push_front(&head, 80);


    print_list(head);
    printf("---------------\n");

    pop_top(&head);
    pop_top(&head);

    print_list(head);
    printf("---------------\n");

    pop_end(&head);
    pop_end(&head);

    print_list(head);    
    printf("---------------\n");

    remove_element(&head,100);
    remove_element(&head,3);

    print_list(head);
    delete_list(&head);
    print_list(head);
    return 0;
}
