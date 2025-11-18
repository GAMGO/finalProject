package com.finalproject.backend.service;

import java.util.Collections;

import org.springframework.stereotype.Service;

import com.google.api.client.googleapis.auth.oauth2.GoogleIdToken;
import com.google.api.client.googleapis.auth.oauth2.GoogleIdTokenVerifier;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.api.client.json.jackson2.JacksonFactory;

@Service
public class GoogleAuthService {

  private final static String CLIENT_ID = "VITE_GOOGLE_CLIENT_ID";

  public static GoogleIdToken.Payload verify(String idTokenString) throws Exception {
    GoogleIdTokenVerifier verifier = new GoogleIdTokenVerifier.Builder(
        new NetHttpTransport(), JacksonFactory.getDefaultInstance())
        .setAudience(Collections.singletonList(CLIENT_ID))
        .build();

    GoogleIdToken idToken = verifier.verify(idTokenString);
    if (idToken != null) {
      return idToken.getPayload();
    } else {
      throw new IllegalArgumentException("Invalid ID Token");
    }
  }
}