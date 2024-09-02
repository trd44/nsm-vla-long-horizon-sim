

(define (domain tabletop)
    (:requirements :strips :typing)
    (:types
        robot holdable opennable support - object
        gripper - robot
        mug coffee-pod coffee-machine-lid - holdable
        container - opennable
        table - support
        coffee-pod-holder - container
    )

    ;; Define predicates
    (:predicates
        (holding ?obj - holdable) ; whether the gripper is holding an object. If true, `free gripper` should be false.
        (open ?container - opennable)
        (free ?gripper - gripper) ; whether the gripper is not holding anything. If true, `holding` should be false.
        (on ?obj - holdable ?support - support)
        (in ?obj - holdable ?container - container)
        (under ?obj1 - object ?obj2 - object)
    )
    
    ;; Define actions using the predicates given
    (:action pick-up
        :parameters (?obj - holdable ?table - table ?gripper - gripper)
        :precondition (and (free ?gripper) (on ?obj ?table))
        :effect (and (holding ?obj) (not (on ?obj ?table)) (not (free ?gripper)))
    )

    (:action release-holding
        :parameters (?held - holdable ?gripper - gripper)
        :precondition (and (not (free ?gripper)) (holding ?held))
        :effect (and (free ?gripper) (not (holding ?held)))
    )
    
    (:action open-coffee-pod-holder
        :parameters (?holder - coffee-pod-holder ?lid - coffee-machine-lid ?gripper - gripper)
        :precondition (and (not (open ?holder)) (free ?gripper))
        :effect (and (open ?holder) (free ?gripper))
    )

    (:action place-pod-in-holder
        :parameters (?pod - coffee-pod ?holder - coffee-pod-holder ?gripper - gripper)
        :precondition (and (holding ?pod) (open ?holder) (not (free ?gripper)))
        :effect (and (not (holding ?pod)) (in ?pod ?holder) (free ?gripper))
    )

    (:action close-coffee-pod-holder
        :parameters (?holder - coffee-pod-holder ?lid - coffee-machine-lid ?gripper - gripper)
        :precondition (and (open ?holder) (free ?gripper))
        :effect (and (not (open ?holder)) (free ?gripper))
    )

    (:action place-mug-under-holder
        :parameters (?mug - mug ?holder - coffee-pod-holder ?table - table ?gripper - gripper)
        :precondition (and (on ?mug ?table) (free ?gripper))
        :effect (and (under ?mug ?holder) (not (on ?mug ?table)) (free ?gripper))
    )

)