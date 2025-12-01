// src/main/java/org/iclass/route/dto/RoutePoint.java
package org.iclass.route.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class RoutePoint {
    private double lat;
    private double lng;
}
