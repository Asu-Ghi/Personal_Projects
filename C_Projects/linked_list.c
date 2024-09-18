#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define max_array = 256

typedef struct node {
    int val;
    struct node* next;
 } node_t;


void print_list(node_t* head) {

    node_t * current = head;

    while (current!=NULL) {
        printf("%d\n", current -> val);
        current = current -> next;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("command usage %s %s", argv[0], "data");
    }

    node_t * head = NULL;
    head = (node_t *) malloc(sizeof(node_t));
    if (head == NULL) {
        return 1;
    }
    
    head -> val = 1;
    head-> next = (node_t *) malloc(sizeof(node_t));
    head -> next -> val = 2

    print_list(head);
    return 0;
}
