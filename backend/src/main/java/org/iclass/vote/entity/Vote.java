package org.iclass.vote.entity;

import org.iclass.customer.entity.CustomersEntity;
import org.iclass.polls.entity.Poll;
import org.iclass.polls.entity.PollOption;
import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;

@Getter 
@Setter 
@NoArgsConstructor 
@AllArgsConstructor 
@Builder
@Entity
@Table(name = "vote",
       uniqueConstraints = {
         @UniqueConstraint(name = "uq_vote_once", columnNames = {"poll_idx", "user_idx"})
       },
       indexes = {
         @Index(name = "ix_vote_poll", columnList = "poll_idx"),
         @Index(name = "ix_vote_user", columnList = "user_idx"),
         @Index(name = "ix_vote_option", columnList = "option_idx")
       })
public class Vote {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "idx")
    private Long idx;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "poll_idx", foreignKey = @ForeignKey(name = "fk_vote_poll"))
    private Poll poll;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_idx", foreignKey = @ForeignKey(name = "fk_vote_user"))
    private CustomersEntity user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "option_idx", foreignKey = @ForeignKey(name = "fk_vote_option"))
    private PollOption option;

    @Column(name = "created_at")
    private LocalDateTime createdAt;
}
