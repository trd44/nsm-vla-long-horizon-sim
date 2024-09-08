

(define (domain coffee)
    (:requirements :strips :typing)
    (:types
        mug coffee-pod coffee-pod-holder coffee-machine-lid drawer table gripper - object
    )

    ;; Define predicates
    (:predicates
        (can-hold ?obj - object)
        (can-open ?obj - object)
        (can-contain ?container - object ?obj - object)
        (on-table ?obj - object ?table - table)
        (holding ?obj - object) ; whether the gripper is holding an object. If true, `free gripper` should be false.
        (in ?obj - object ?container - object)
        (open ?container - object)
        (free ?gripper - gripper) ; whether the gripper is not holding anything. If true, `holding` should be false.
        (under ?bottom - object ?top - object)
    )
    
    ;; Define actions using the predicates given
    (:action pick-up-tabletop
        :parameters (?obj - object ?table - table ?gripper - gripper) 
        :precondition (and (on-table ?obj ?table) (can-hold ?obj) (free ?gripper)) 
        :effect (and (holding ?obj) (not (on-table ?obj ?table)) (not (free ?gripper))) 
    )
    
    (:action open-coffee-pod-holder
        :parameters (?holder - coffee-pod-holder ?lid - coffee-machine-lid ?gripper - gripper)
        :precondition (and (not (open ?holder)) (can-open ?holder) (free ?gripper))
        :effect (open ?holder)
    )

    (:action close-coffee-pod-holder
        :parameters (?holder - coffee-pod-holder ?lid - coffee-machine-lid ?gripper - gripper)
        :precondition (and (open ?holder) (free ?gripper))
        :effect (not (open ?holder))
    )

    (:action place-pod-in-holder
        :parameters (?pod - coffee-pod ?holder - coffee-pod-holder ?gripper - gripper)
        :precondition (and (holding ?pod) (open ?holder) (can-contain ?holder ?pod) (not (free ?gripper)))
        :effect (and (not (holding ?pod)) (in ?pod ?holder) (free ?gripper))
    )


    (:action place-mug-under-holder
        :parameters (?mug - mug ?holder - coffee-pod-holder ?gripper - gripper)
        :precondition (holding ?mug)
        :effect (and (under ?mug ?holder) (not (holding ?mug)) (free ?gripper))
    )

)