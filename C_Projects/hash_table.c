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

int main() {
     printf("Jacob => %u\n", hash("Jacob")); 
     printf("Natalie => %u\n", hash("Natalie")); 
     printf("Mark => %u\n", hash("Mark")); 
     printf("John => %u\n", hash("John")); 
     return 0;
}





























