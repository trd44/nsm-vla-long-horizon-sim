(define (domain minimal)
    (:requirements :strips :typing)

    (:types
        object
        robot
        holdable - object
        opennable - object
        support - object
        gripper - robot
        mug coffee-pod coffee-machine-lid - holdable
        container - opennable
        table - support
        coffee-pod-holder - container
    )

    (:predicates
        (on ?obj - object ?surface - object)
        (clear ?surface - object)
    )

    (:action place-on-table
        :parameters (?obj - object ?table - table)
        :precondition (clear ?table)
        :effect (and (on ?obj ?table) (not (clear ?table)))
    )
)
