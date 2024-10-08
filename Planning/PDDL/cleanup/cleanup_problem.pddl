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
        (can-pick-up mug1)
        (can-pick-up block1)
        (can-contain drawer1 block1)
        (can-contain drawer1 mug1)
        (can-contain mug1 block1)
        (in block1 mug1)
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