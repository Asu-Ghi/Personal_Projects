#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
 
#define MAX_NAME 256
#define TABLE_SIZE 10


typedef struct {
	char name[MAX_NAME];
	int age;
} person;


person * hash_table[TABLE_SIZE];

unsigned int hash(char* name) {
	int length = strnlen(name, MAX_NAME);
	unsigned int hash_value = 0;
	for (int i = 0; i < length; i++) {
		// Adds ascii value of char to hash_value
		hash_value += name[i];
		// adds more random to hash_value and ensures it doesnt exceed table size
		hash_value = hash_value * name[i] % TABLE_SIZE;
		
	}
	return hash_value;
}

void init_hash_table(){

	for (int i = 0; i < TABLE_SIZE; i++) {
		hash_table[i] = NULL;
	}
	// table is empty
}

void print_table() {
	printf("Start \n");
	for (int i = 0; i < TABLE_SIZE; i++) {;
		if (hash_table[i] == NULL) {
			printf("\t%i\t ---\n", i);
		}

		else {
			printf("\t%i\t%s\n", i, hash_table[i]->name);				
		}
	}	
	printf("End \n");
}

bool insert_hash_table(person* p) {
	if (p == NULL) return false;
	
	int index = hash(p->name);
	
	if (hash_table[index] != NULL) {
		return false;	
	}
	
	hash_table[index] = p;
	return true;
}

// Find a person by their name
person *hash_table_lookup(char *name ) {
	int index = hash(name);
	if (hash_table[index] != NULL &&
		strncmp(hash_table[index]->name, name, TABLE_SIZE) == 0) {
		return hash_table[index];
	} else {
		return NULL;
	}
}

person *hash_table_delete(char *name) {
	return NULL;
}

int main() {
     init_hash_table();
     print_table();
     person jacob = {.name="Jacob", .age=17};
     person natalie = {.name="Natalie", .age=20};
     person mpho = {.name="mpho", .age=200 };
     person mark = {.name="mark", .age= 21};

     insert_hash_table(&jacob);
     insert_hash_table(&natalie);
     insert_hash_table(&mpho);
     insert_hash_table(&mark);
     person *temp = hash_table_lookup("mpho");

     if (temp == NULL) {
	printf("Person not found! \n");
	
     } else {
	printf("Found %s \n ", temp->name);
     }
	 
     print_table();
     return 0;
}





























