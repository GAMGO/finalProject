package org.iclass.store.controller;

import lombok.RequiredArgsConstructor;

import org.iclass.store.dto.StoreChangeRejectRequest;
import org.iclass.store.entity.StoreChangeRequestResponse;
import org.iclass.store.service.StoreChangeAdminService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/admin/store-changes")
@RequiredArgsConstructor
public class StoreChangeAdminController {

    private final StoreChangeAdminService adminService;

    @GetMapping("/pending")
    public ResponseEntity<List<StoreChangeRequestResponse>> listPending() {
        return ResponseEntity.ok(adminService.listPending());
    }

    @PostMapping("/{id}/approve")
    public ResponseEntity<Void> approve(@PathVariable Long id) {
        Long adminId = 1L; // TODO: 인증된 관리자 ID
        adminService.approveChange(id, adminId);
        return ResponseEntity.noContent().build();
    }

    @PostMapping("/{id}/reject")
    public ResponseEntity<Void> reject(@PathVariable Long id,
                                       @RequestBody StoreChangeRejectRequest req) {
        Long adminId = 1L; // TODO: 인증된 관리자 ID
        adminService.rejectChange(id, adminId, req.getReason());
        return ResponseEntity.noContent().build();
    }
}

