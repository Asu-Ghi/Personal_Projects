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

// creates alphabet table with all letters a-z
// init counts as 0.
void create_alphabet_table(letter* letter_table[]) {
    int indx = 0;
    // append letter_table with letters a - z
    for (int i = 97; i < 123; i++) {
        letter* temp = (letter*) malloc(sizeof(letter*));
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


// main function
int main(int argc, char** argv) {
    // check command line arguments
    if (argc < 2) {
        printf("Command usage %s %s %s.\n", argv[0], "word_one", 
        "word_two");
        return 1;
    }

    // create letter table -> tracks count and max possible count of each letter in a word.
    letter* table[27];
    // create and print table for all letter values
    create_alphabet_table(table);
    // print_alphabet_table(table);
    
    // create variables 
    char* word_one = argv[1];
    char* word_two = argv[2];
    int length = strlen(word_one);

    // check if stringes equal each other
    if (strlen(word_one) != strlen(word_two)) {
        printf("error : hidden word is not the same length as the guess word\n");
        return 1;
    }
    
    // create hint array size length
    char green_hints[length];
    char yellow_hints[length];
    char red_hints[length];
    char big_hints[length];
    // fill hints with placeholder
    memset(green_hints, '-', length);
    memset(yellow_hints, '-', length);
    memset(red_hints, '-', length);
    memset(big_hints, '-', length);

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
    // print hints
    printf("Green hints: %s\n", green_hints);
    printf("Yellow hints: %s\n", yellow_hints);
    printf("Red hints: %s\n", red_hints);
    printf("Big hints: %s\n", big_hints);


}