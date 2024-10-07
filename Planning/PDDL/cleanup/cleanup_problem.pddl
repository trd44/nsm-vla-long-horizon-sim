(define (problem cleanup_problem)
    (:domain cleanup)

    (:objects
        block1 - block
        mug1 - mug
        drawer1 - drawer
        table1 - table
        gripper1 - gripper
    )

    (:init
        (free gripper1)
        (can-hold block1)
        (can-hold mug1)
        (can-open drawer1)
        (can-contain drawer1 block1)
        (can-contain drawer1 mug1)
        (can-contain mug1 block1)
        (on-table block1 table1)
        (on-table mug1 table1)
        (on-table drawer1 table1)
    )

    (:goal
        (and
            (in block1 drawer1)
            (not (open drawer1))
        )
    )
)