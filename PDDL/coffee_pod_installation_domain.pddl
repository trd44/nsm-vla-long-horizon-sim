(define (domain coffee_pod_installation_domain)
    (:requirements :strips :typing)
    
    ;; Define types
    (:types
        object
        holdable - object
        opennable - object
        nothing mug coffee-pod - holdable
        container - opennable
        container - containable
        coffee-pod-holder - container
    )
    
    ;; Define predicates
    (:predicates
        (holding ?obj - holdable)
        (open ?container - opennable)
        (pod-installed ?obj - coffee-pod)
        (in ?obj - holdable ?container - containable)
        ）
    )
    
    ;; Define actions
    (:action pick-up-pod
        :parameters (?pod - coffee-pod ?hand - hand)
        :precondition (and (empty ?hand) (on-table ?pod))
        :effect (and (not (empty ?hand)) (holding ?pod))
    )

    (:action open-machine
        :parameters (?machine - coffee-machine)
        :precondition (and (closed ?machine))
        :effect (and (not (closed ?machine)) (open ?machine))
    )

    (:action place-pod-in-machine
        :parameters (?pod - coffee-pod ?machine - coffee-machine ?hand - hand)
        :precondition (and (holding ?pod) (open ?machine))
        :effect (and (not (holding ?pod)) (in ?pod ?machine) (empty ?hand))
    )

    (:action close-machine
        :parameters (?machine - coffee-machine)
        :precondition (and (open ?machine))
        :effect (and (not (open ?machine)) (closed ?machine) (pod-installed ?pod))
    )
    
    
)
