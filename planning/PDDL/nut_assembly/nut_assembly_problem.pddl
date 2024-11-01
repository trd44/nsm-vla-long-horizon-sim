(define (problem nut_assembly)
    (:domain nut_assembly)
    (:objects
        round-nut1 - round-nut
        square-nut1 - square-nut
        round-peg1 - round-peg
        square-peg1 - square-peg
        table1 - table
        gripper1 - gripper
    )

    (:init
        (small-enough-for-gripper-to-pick-up round-nut1 gripper1)
        (small-enough-for-gripper-to-pick-up square-nut1 gripper1)
        (shapes-match round-nut1 round-peg1)
        (shapes-match square-nut1 square-peg1)
        (free gripper1)
        (on-peg round-nut1 round-peg1)
        (on-peg square-nut1 round-peg1)
        (directly-on-table round-peg1 table1)
        (directly-on-table square-peg1 table1)
        (directly-on-table round-nut1 table1)
    )

    (:goal 
        (and
            (on-peg square-nut1 square-peg1)
        )
    )
)