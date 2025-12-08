package org.iclass.polls.entity;

import jakarta.persistence.*;
import lombok.*;

@Getter 
@Setter 
@NoArgsConstructor 
@AllArgsConstructor 
@Builder
@Entity
@Table(name = "polls_options",
       indexes = {
         @Index(name = "ix_options_poll", columnList = "poll_idx")
       })
public class PollOption {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "idx")
    private Long idx;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "poll_idx", foreignKey = @ForeignKey(name = "fk_options_poll"))
    private Poll poll;

    @Column(name = "description", length = 1024)
    private String description;

    @Column(name = "order")
    private Long orderNo;
}

