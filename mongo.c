#include <stdio.h>
#include <string.h>
#include <curl/curl.h>
#include <unistd.h>

void insert(unsigned long int UTC_Time, float lat, float lng, float spd) {
    unsigned int hour, min, sec;
    char time_str[9];  
    hour = (UTC_Time / 10000);
    min = (UTC_Time % 10000) / 100;
    sec = (UTC_Time % 10000) % 100;
    
    hour = hour + 5;  
    if (hour >= 24) {
        hour -= 24;  
    }
    
    min = min + 30;
    if (min > 59) {
        min -= 60;  
    }

    sprintf(time_str, "%02u:%02u:%02u", hour, min, sec);

    CURL *curl;
    CURLcode res;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if(curl) {
        const char *url = "https://ap-south-1.aws.data.mongodb-api.com/app/data-xnakx/endpoint/data/v1/action/insertOne";
        const char *api_key = "o87MBXtxmrtHBATmbQHxDrq2cYGmjxhfmh7szC8fn8C22qFJ2i1My4bGUEmtdJQi";
        char json_payload[1024];

        sprintf(json_payload, 
                "{"
                "\"dataSource\": \"Cluster0\","
                "\"database\": \"jan23\","
                "\"collection\": \"test3\","
                "\"document\": {"
                "\"time\": \"%s\","
                "\"latitude\": %.6f,"
                "\"longitude\": %.6f,"
                "\"speed\": %.2f"
                "}"
                "}", time_str, lat, lng, spd);

        struct curl_slist *headers = NULL;
        char api_key_header[256];
        sprintf(api_key_header, "api-key: %s", api_key);

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
        unsigned long int UTC_Time = 105300;
        float lat = 27.123;
        float lng = 77.234;
        float spd = 2.34;

        insert(UTC_Time, lat, lng, spd);
        sleep(2);
    }

    return 0;
}

