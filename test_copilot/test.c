#include <stdio.h>
#include <stdlib.h>
#include <string.h>
char* crypt_vigenere(char* plaintext, char* keyword) {
    char* ciphertext = malloc(sizeof(char) * (strlen(plaintext) + 1));
    int k = 0;
    for (int i = 0; i < strlen(plaintext); i++) {
        if (plaintext[i] >= 'a' && plaintext[i] <= 'z') {
            ciphertext[i] =
                ((plaintext[i] - 'a' + keyword[k] - 'a') % 26) + 'a';
            k++;
            if (k == strlen(keyword)) {
                k = 0;
            }
        } else {
            ciphertext[i] = plaintext[i];
        }
    }
    ciphertext[strlen(plaintext)] = '\0';
    return ciphertext;
}

char* decrypt_vigenere(char* ciphertext, char* keyword) {
    char* plaintext = malloc(sizeof(char) * (strlen(ciphertext) + 1));
    int k = 0;
    for (int i = 0; i < strlen(ciphertext); i++) {
        if (ciphertext[i] >= 'a' && ciphertext[i] <= 'z') {
            plaintext[i] = ((ciphertext[i] - 'a' - keyword[k] + 26) % 26) + 'a';
            k++;
            if (k == strlen(keyword)) {
                k = 0;
            }
        } else {
            plaintext[i] = ciphertext[i];
        }
    }
    plaintext[strlen(ciphertext)] = '\0';
    return plaintext;
}

int main() {
    char* plaintext = "abcdefghijklmnopqrstuvwxyz";
    char* keyword = "cipher";
    char* ciphertext = crypt_vigenere(plaintext, keyword);
    printf("%s\n", ciphertext);
    return 0;
}