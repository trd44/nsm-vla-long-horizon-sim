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
        (can-pick-up round-nut1)
        (can-pick-up square-nut1)
        (match round-nut1 round-peg1)
        (match square-nut1 square-peg1)
        (free gripper1)
        (on round-nut1 round-peg1)
        (on square-nut1 round-peg1)
        (on-table round-peg1 table1)
        (on-table square-peg1 table1)
        (on-table round-nut1 table1)
    )

    (:goal 
        (and
            (on square-nut1 square-peg1)
        )
    )
)