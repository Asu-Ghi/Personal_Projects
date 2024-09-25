#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

// handles repeat letters
typedef struct {
    char value;
    int count;
    int count_allowed;
} letter;

typedef struct {
    char val[6];
} words;



// creates alphabet table with all letters a-z
// init counts as 0.
void create_alphabet_table(letter* letter_table[]) {
    int indx = 0;
    // append letter_table with letters a - z
    for (int i = 97; i < 123; i++) {
        letter* temp = (letter*) malloc(sizeof(letter));
        temp -> value = (char)i;
        temp -> count = 0;
        temp -> count_allowed =0;
        letter_table[indx] = temp; 
        indx ++;
    }
}

// return count of letter in table
letter* lookup_alphabet_table(letter* letter_table[], char letter) {
    // iterate through table
    for (int i =0; i < 26; i++) {
        if (letter_table[i] -> value == letter) {
            return letter_table[i];
        }
    }

    return NULL; //couldnt find letter in table
}
//debuging method
void print_alphabet_table (letter* letter_table[]) {
    // print out letter table
    for (int i = 0; i < 26; i++) {
        printf("Letter: %c Count: %d \n", letter_table[i] -> value, 
        letter_table[i] -> count);
    }
}

// updates letter table with max counts for letters in word given
void get_counts(letter** letter_table, char* word) {
    int length = strlen(word);
    for (int i=0; i < length; i++) {
        letter* temp = lookup_alphabet_table(letter_table, word[i]);
        temp -> count_allowed++;
    }
}

// function for evaluation green hints first, as to not overlap with yellow.
void get_green (letter** letter_table, char* word_one, char* word_two, char* green_hints) {
    int length = strlen(word_one);
    // check to see if a letter is in the same spot
    for (int i =0; i < length; i++) {
        if (word_one[i] == word_two[i]) {
            lookup_alphabet_table(letter_table, word_one[i]) -> count += 1;
            green_hints[i] = 'g';
        }
    }
}

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

// create hints
char* create_hints(char* word_one, char* word_two) {
    // create letter table -> tracks count and max possible count of each letter in a word.
    letter* table[27];
    // create and print table for all letter values
    create_alphabet_table(table);

    int length = strlen(word_one);
    // create hint array size length
    char green_hints[length];
    char yellow_hints[length];
    char big_hints[length];

    // fill hints with placeholder
    memset(green_hints, '-', length);
    green_hints[length] = '\0'; 
    memset(yellow_hints, '-', length);
    yellow_hints[length] = '\0'; 
    memset(big_hints, '-', length);
    big_hints[length] = '\0'; 

    // get max letter counts possible for word_one, and update the table.
    get_counts(table, word_one);

    // evaluate green hints first.
    get_green(table, word_one, word_two, green_hints);

// evaluate yellow and then red hints.
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {

            // find letter value in alphabet count table
            letter* temp = lookup_alphabet_table(table, word_one[i]);

            // check to see if a letter is in the word, but not the right spot. 
            // check table to see if letter has repeat apperances.
            // check to see if spot already taken by a green hint.
            // incriment hash table for letter if both conditions are passed.
            if (word_one[i] == word_two[j] && 
            temp->count_allowed > temp->count &&
            green_hints[j] == '-') {
                lookup_alphabet_table(table, word_one[i]) -> count += 1;
                yellow_hints[j] = 'y';
            }
        }
    }

    // free up letter table
    for (int i =0; i < 26; i ++) {
        free(table[i]);
    }

    // append all hints together.
    for(int i =0; i < length; i++) {
        if(green_hints[i] != '-') {
            big_hints[i] = green_hints[i];
        } 
        else if(yellow_hints[i] != '-') {
            big_hints[i] = yellow_hints[i];
        }
        else {
            big_hints[i] = 'r';
        }
    }
    char* final_hints = (char*)malloc(100 * sizeof(char)); // Allocate memory on the heap
    final_hints = big_hints;
    return final_hints;
}

int main(int argc, char** argv) {

    // handles command line 
    if (argc < 2) {
        printf("command usage %s %s %s\n", argv[0], "word", "hints");
        return 1;
    }
    // test
    char* word_one = argv[1];
    char* valid_hints = argv[2];

    // read in data set
    int data_length = 2309;
    words data_set[data_length];    
    read_data(data_set, data_length);
    // print_data(data_set, data_length);
    int indicies[100] = {0}; 
    int in_count = 0;
    for(int i =0 ; i < data_length; i++) {
        char * test_hints = create_hints(word_one, data_set[i].val);
        if (strcmp(test_hints, valid_hints) == 0) {
            printf("%s\n", data_set[i].val);
        }
    }








}