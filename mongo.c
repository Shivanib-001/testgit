#include <stdio.h>
#include <string.h>
#include <curl/curl.h>
#include <unistd.h>

void insert() {
    CURL *curl;
    CURLcode res;

    // Initialize libcurl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if(curl) {
        const char *url = "https://ap-south-1.aws.data.mongodb-api.com/app/data-xnakx/endpoint/data/v1/action/insertOne";
        const char *api_key = "o87MBXtxmrtHBATmbQHxDrq2cYGmjxhfmh7szC8fn8C22qFJ2i1My4bGUEmtdJQi";
        const char *json_payload = 
        "{"
        "\"dataSource\": \"Cluster0\","
        "\"database\": \"learn-data-api\","
        "\"collection\": \"testCollection\","
        "\"document\": "
        "{"
        "\"latitude\":64.678,"
        "\"longitude\":73.345"
        "}"
        "}";

        struct curl_slist *headers = NULL;
        char api_key_header[256];
        snprintf(api_key_header, sizeof(api_key_header), "api-key: %s", api_key);

        // Set headers for the request
        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, api_key_header);

        // Set curl options
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload);

        // Perform the request
        res = curl_easy_perform(curl);

        if(res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        } else {
            printf("Document inserted successfully!\n");
        }

        // Clean up
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }

    curl_global_cleanup();
}

int main() {
    while (1) {
        insert();
        sleep(2);  // Sleep for 2 seconds
    }

    return 0;
}

