(define (domain coffee-pod-installation)
    (:requirements :strips :typing)
    
    ;; Define types
    (:types
        object
        robot
        holdable - object
        opennable - object
        support - object
        gripper - robot
        mug coffee-pod - holdable
        container - opennable
        table dispenser - support
        coffee-machine - container
    )
    
    ;; Define predicates
    (:predicates
        (holding ?obj - holdable)
        (open ?container - opennable)
        (free ?gripper - gripper)
        (on ?obj - holdable ?support - support)
        (in ?obj - holdable ?container - container)
    )
    
    ;; Define actions using the predicates given
    (:action pick-up-pod
        :parameters (?pod - coffee-pod ?gripper - gripper)
        :precondition (and (free ?gripper) (in ?pod drawer) (open drawer))
        :effect (and (holding ?pod) (not (in ?pod drawer)) (not (free ?gripper)))
    )

    (:action open-machine
        :parameters (?machine - opennable ?gripper - gripper)
        :precondition (and (not (open ?machine)) (free ?gripper))
        :effect (and (open ?machine) (free ?gripper))
    )

    (:action place-pod-in-machine
        :parameters (?pod - coffee-pod ?machine - container ?gripper - gripper)
        :precondition (and (holding ?pod) (open ?machine))
        :effect (and (not (holding ?pod)) (in ?pod ?machine) (free ?gripper))
    )

    (:action close-machine
        :parameters (?machine - opennable ?gripper - gripper)
        :precondition (and (open ?machine) (free ?gripper))
        :effect (and (not (open ?machine)) (free ?gripper))
    )

    (:action open-drawer
        :parameters (?drawer - opennable ?gripper - gripper)
        :precondition (and (not (open ?drawer)) (free ?gripper))
        :effect (and (open ?drawer) (free ?gripper))
    )

    (:action close-drawer
        :parameters (?drawer - opennable ?gripper - gripper)
        :precondition (and (open ?drawer) (free ?gripper))
        :effect (and (not (open ?drawer)) (free ?gripper))
    )

    (:action place-cup-on-dispenser
        :parameters (?cup - mug ?dispenser - dispenser ?gripper - gripper)
        :precondition (and (holding ?cup))
        :effect (and (on ?cup ?dispenser) (not (holding ?cup)) (free ?gripper))
    )

    (:action pick-up-cup
        :parameters (?cup - mug ?gripper - gripper)
        :precondition (and (free ?gripper) (on ?cup table))
        :effect (and (holding ?cup) (not (on ?cup table)) (not (free ?gripper)))
    )
    
)
