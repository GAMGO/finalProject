package org.iclass.finalproject.customer.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class LogoutResponse {
    private String message;
}
