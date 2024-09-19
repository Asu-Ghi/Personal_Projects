#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#define MAX_WORD_LENGTH 6
// data type for words.
typedef struct {
    char val[MAX_WORD_LENGTH];
} words;

// data type for alphabet
typedef struct {
    char val[2];
    char identifier[2];
    int count;
    int index;
} letter;

// create alphabet
void create_alphabet_table(letter* letter_table[]) {
    for (int i = 0; i < 26; i++) {
        letter_table[i] = malloc(sizeof(letter)); // Allocate memory
        if (letter_table[i] != NULL) {
            letter_table[i]->val[0] = 'a' + i; // Assign letters a-z
            letter_table[i]->val[1] = '\0';     // Null-terminate the string
            letter_table[i]->count = 0;
            letter_table[i]->identifier[0] = '-';
            letter_table[i]->identifier[1] = '\0';
            letter_table[i]->index = -1;

        }
    }
}

//debuging method
void print_alphabet_table(letter* letter_table[]) {
    // Print out letter table
    for (int i = 0; i < 26; i++) {
        // Check if letter_table[i] is not NULL
        if (letter_table[i] != NULL) {
            printf("Letter: %s Count: %d Identifier: %s Index: %d \n", letter_table[i]->val, 
            letter_table[i]->count, letter_table[i]->identifier, letter_table[i]->index);
        } else {
            printf("Letter table entry %d is NULL\n", i);
        }
    }
}

// lookup letter in alphabet table by value
letter* lookup_alphabet_table(letter* letter_table[], char letter) {
    // iterate through table
    for (int i =0; i < 26; i++) {
        if (letter_table[i] -> val[0] == letter) {
            return letter_table[i];
        }
    }
    return NULL; //couldnt find letter in table
}


// update alphabet_table
void update_alphabet_table(letter* letter_table[], char* word, char* hints) {
    for(int i = 0; i < strlen(word); i++) {
        letter* temp = lookup_alphabet_table(letter_table, word[i]);
        temp -> count++;
        temp -> identifier[0] = hints[i];
        temp-> index = i;
    }
}

//

// read in data
void read_data(words* data, int length) {
    for (int i = 0; i < length; i++) {
        if (scanf("%s", data[i].val) != 1) { // Read word into val
            printf("Exception in reading data \n");
            exit(1);
        }
    }
}
// debug method
void print_data(words* data, int length) {
    for (int i = 0; i < length; i++) {
        printf("%s\n", data[i].val);
    }
}

int main(int argc, char** argv) {

    // handles command line 
    if (argc < 2) {
        printf("command usage %s %s %s\n", argv[0], "word", "hints");
        return 1;
    }
    // test
    char* word = "words";
    char* hints = "ggggg";
    // read in data set
    int data_length = 2309;
    words data_set[data_length];    
    read_data(data_set, data_length);
    // print_data(data_set, data_length);

    // create alphabet table
    letter* alphabet[27];
    create_alphabet_table(alphabet);
    print_alphabet_table(alphabet);
    update_alphabet_table(alphabet, word, hints);
    printf("-----------\n");
    print_alphabet_table(alphabet);








}