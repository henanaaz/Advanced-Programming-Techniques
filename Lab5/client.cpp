/*
Author: Hena Naaz
Class: ECE6122
Last Date Modified: 20 Nov
Description:
The purpose of this file is to create clients for TCP connections.
*/

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>
using namespace std;

/*
    check the port number passed in the console argument is valid or not
    and return port number if it valid else it will return 0

    @return int
*/
int getPortNumber(char* portNumber) {
    for(int i=0; portNumber[i] != '\0'; i++) {
        if(portNumber[i]<'0' || portNumber[i]>'9') {
            return 0;
        }
    }

    return atoi(portNumber);
}


/*
    check the ip address passed in the console argument is valid or not
    and return ip address  if it valid else it will return 0

    @return string
*/
string getIPAddress(char *ipAddress) {

    // checking ip address entered is correct or not
    for(int i=0; ipAddress[i] != '\0'; i++) {
        if(!(ipAddress[i]>='a' && ipAddress[i]<='z')) {
            return "0";
        }
    }

    return strcpy(ipAddress, "localaddress") ? "127.0.0.1" : "0";
}


int main(int argc, char *argv[])
{
    struct sockaddr_in clientAddress;
    int clientSocket;
    char msg[500];
    char res[600];
    char ip[INET_ADDRSTRLEN];

    // console arguments are incorrect
    if(argc > 3) {
        printf("Too many arguments");
        exit(1);
    }

    int portNumber = getPortNumber(argv[2]);
    char ipAddress[100];
    strcpy(ipAddress, getIPAddress(argv[1]).c_str());

    // checking port number is in the valid range or not
    if( portNumber < 61000 || portNumber > 65535) {
        printf("Invalid command line argument detected: %d \nPlease check your values and press any key to end the program!", portNumber);
        exit(1);
    }

    // checking ip address is valid or not
    if(strcmp(ipAddress, "0") == 0) {
        printf("Invalid command line argument detected: %s \nPlease check your values and press any key to end the program!", argv[1]);
        exit(1);
    }

    clientSocket = socket(AF_INET,SOCK_STREAM, 0);


    // initializing variables of sockaddr_in structure
    for(int i=0; i<8; i++) {
        clientAddress.sin_zero[i] = '\0';
    }

    clientAddress.sin_family = AF_INET;
    clientAddress.sin_port = htons(portNumber);
    clientAddress.sin_addr.s_addr = inet_addr(ipAddress);

    // creating TCP connection with the server
    if(connect(clientSocket, (struct sockaddr *) &clientAddress, sizeof(clientAddress)) < 0) {
        printf("Failed to connect to the server at %s on %d. \nPlease check your values and press any key to end program!", ipAddress, portNumber);
        exit(1);
    }
    
    inet_ntop(AF_INET, (struct sockaddr *) &clientAddress, ipAddress, INET_ADDRSTRLEN); 

    printf("Please enter a message: ");


    // initializing character array which reads from the console and transmit over the network
    for(int i=0; i<500; i++) {
        res[i] = '\0';
        msg[i] = '\0';
    }
    
    // continously reading the message from the console
    while(fgets(msg, 500, stdin) != NULL) {

        for(int i=0; i<500; i++) {
            res[i] = msg[i];
        }

        // on entering quit, terminating the client connection
        if(std::string(msg) == "quit\n") {
            printf("connection close");
            exit(1);
        }

        int len = write(clientSocket, msg, strlen(res));
        
        // checking whether message is sent or not
        if(len < 0) {
            printf("message not sent");
            exit(1);
        }
        
        memset(msg,'\0',sizeof(msg));
        printf("Please enter a message: ");
    }
    
    close(clientSocket);
}