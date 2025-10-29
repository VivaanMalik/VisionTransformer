#include <stdio.h>
#include <string.h>
#include "../include/test.h"

void print(const char *text) {
    for (int i =0; i < strlen(text); i++) {
        printf("%c", text[i]);
    }
}